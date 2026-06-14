#!/usr/bin/env python3
"""PyTorch CNN for Freesound Audio Tagging 2019.

Trains on log-mel spectrogram crops with SpecAugment, Focal Loss, and Mixup.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in this environment. Install it with:\n"
        "  python3 -m pip install -r requirements_cnn.txt\n"
        "Then rerun this script."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
BAD_CURATED_FILES = {
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
    "1d44b0bd.wav",
}
BAD_NOISY_FILES = {
    "0e42ddd2.wav",
    "0f0cab1d.wav",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vocabulary(root: Path) -> list[str]:
    vocab = pd.read_csv(root / "meta" / "vocabulary.csv", header=None, names=["idx", "label"])
    return vocab.sort_values("idx")["label"].tolist()


def split_labels(value: str) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def rows_with_existing_audio(df: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    exists = df["fname"].map(lambda name: (audio_dir / name).exists())
    return df.loc[exists].copy()


def load_training_rows(root: Path, include_noisy: bool) -> pd.DataFrame:
    curated = pd.read_csv(root / "meta" / "train_curated_post_competition.csv")
    curated = curated[~curated["fname"].isin(BAD_CURATED_FILES)].copy()
    curated["audio_dir"] = str(root / "dataset" / "train_curated")
    curated = rows_with_existing_audio(curated, root / "dataset" / "train_curated")
    curated["source"] = "curated"
    frames = [curated]

    noisy_dir = root / "dataset" / "train_noisy"
    noisy_meta = root / "meta" / "train_noisy_post_competition.csv"
    if include_noisy and noisy_dir.exists() and noisy_meta.exists():
        noisy = pd.read_csv(noisy_meta)
        noisy = noisy[~noisy["fname"].isin(BAD_NOISY_FILES)].copy()
        noisy["audio_dir"] = str(noisy_dir)
        noisy = rows_with_existing_audio(noisy, noisy_dir)
        noisy["source"] = "noisy"
        frames.append(noisy)

    rows = pd.concat(frames, ignore_index=True)
    if rows.empty:
        raise RuntimeError("No training audio found. Check dataset/train_curated/ and metadata files.")
    return rows


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_fft // 2)
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def read_wav_mono(path: Path, target_sr: int) -> np.ndarray:
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return np.zeros(target_sr, dtype=np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
    return audio


class LogMelExtractor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fb = mel_filterbank(sample_rate, n_fft, n_mels, 20.0, sample_rate / 2.0)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if audio.size < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - audio.size))
        _, _, zxx = stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            boundary=None,
            padded=False,
        )
        power = np.abs(zxx).astype(np.float32) ** 2
        mel = np.maximum(self.fb @ power, 1e-10)
        logmel = np.log(mel)
        mean = logmel.mean()
        std = logmel.std() + 1e-6
        return ((logmel - mean) / std).astype(np.float32)


class AudioTaggingDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        classes: list[str],
        clip_seconds: float,
        mode: str,
        sample_rate: int = 22050,
        n_mels: int = 128,
        tta_crops: int = 1,
        spec_augment: bool = True,
        freq_mask_param: int = 12,
        time_mask_param: int = 24,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.classes = classes
        self.clip_samples = int(sample_rate * clip_seconds)
        self.mode = mode
        self.sample_rate = sample_rate
        self.tta_crops = max(1, tta_crops)
        self.spec_augment = spec_augment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.extractor = LogMelExtractor(sample_rate=sample_rate, n_mels=n_mels)
        self.label_map = self._build_targets() if mode != "test" and "labels" in self.rows.columns else None

    def _build_targets(self) -> np.ndarray:
        mlb = MultiLabelBinarizer(classes=self.classes)
        labels = [split_labels(value) for value in self.rows["labels"]]
        return mlb.fit_transform(labels).astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_audio(self, audio: np.ndarray, crop_index: int = 0) -> np.ndarray:
        if audio.size < self.clip_samples:
            return np.pad(audio, (0, self.clip_samples - audio.size))
        if audio.size == self.clip_samples:
            return audio

        max_start = audio.size - self.clip_samples
        if self.mode == "train":
            start = random.randint(0, max_start)
        elif self.tta_crops <= 1:
            start = max_start // 2
        else:
            start = round(max_start * crop_index / (self.tta_crops - 1))
        return audio[start : start + self.clip_samples]

    def _spec_augment(self, spec: np.ndarray) -> np.ndarray:
        """Frequency + time masking on log-mel spectrogram (SpecAugment).

        `spec` shape is (n_mels, time). Masks are filled with the spec mean.
        """
        n_mels, n_frames = spec.shape
        fill_val = spec.mean()
        # ----- frequency masking -----
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, n_mels - f))
            spec[f0 : f0 + f, :] = fill_val

        # ----- time masking -----
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, n_frames - t))
            spec[:, t0 : t0 + t] = fill_val

        return spec

    def _load_item(self, index: int, crop_index: int = 0) -> torch.Tensor:
        row = self.rows.iloc[index]
        try:
            audio = read_wav_mono(Path(row.audio_dir) / row.fname, self.sample_rate)
        except Exception:
            tqdm.write(f"[warn] skip corrupted file: {row.fname}")
            audio = np.zeros(self.clip_samples, dtype=np.float32)
        audio = self._crop_audio(audio, crop_index)
        if self.mode == "train":
            audio = audio * np.random.uniform(0.75, 1.25)
            audio = np.clip(audio, -1.0, 1.0)
        logmel = self.extractor(audio)
        if self.mode == "train" and self.spec_augment:
            logmel = self._spec_augment(logmel)
        return torch.from_numpy(logmel).unsqueeze(0)

    def __getitem__(self, index: int):
        if self.mode == "test" and self.tta_crops > 1:
            crops = [self._load_item(index, crop_index=i) for i in range(self.tta_crops)]
            x = torch.stack(crops, dim=0)
        else:
            x = self._load_item(index)

        fname = self.rows.iloc[index].fname
        if self.label_map is None:
            return x, fname
        return x, torch.from_numpy(self.label_map[index]), fname


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """ConvBlock with a residual skip-connection. Input/output channels must match."""
    def __init__(self, channels: int, dropout: float, pool: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        out = out + residual
        out = self.act(out)
        out = self.pool(out)
        return self.dropout(out)


class SmallAudioCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout * 0.5),
            ConvBlock(32, 64, dropout),
            ConvBlock(64, 128, dropout),
            ConvBlock(128, 256, dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        avg = torch.mean(x, dim=(2, 3))
        maxv = torch.amax(x, dim=(2, 3))
        return self.head(torch.cat([avg, maxv], dim=1))


class DeepAudioCNN(nn.Module):
    """18-layer CNN: 6 conv blocks + 3 residual blocks, ~15M params."""
    def __init__(self, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        d = dropout
        self.block1 = ConvBlock(1, 64, d * 0.5, pool=True)
        self.block2 = ConvBlock(64, 128, d * 0.5, pool=True)
        self.block3 = ConvBlock(128, 256, d, pool=True)
        self.block4 = ConvBlock(256, 512, d, pool=True)
        self.resblocks = nn.Sequential(
            ResBlock(512, d, pool=False),
            ResBlock(512, d, pool=False),
            ResBlock(512, d, pool=False),
        )
        self.block5 = ConvBlock(512, 512, d, pool=False)
        self.block6 = ConvBlock(512, 768, d, pool=False)
        self.head = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(d),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(d * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.resblocks(x)
        x = self.block5(x)
        x = self.block6(x)
        avg = torch.mean(x, dim=(2, 3))
        maxv = torch.amax(x, dim=(2, 3))
        return self.head(torch.cat([avg, maxv], dim=1))


class FocalBCEWithLogitsLoss(nn.Module):
    """Multi-label Focal Loss for handling class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_weight * focal_weight * bce_loss).mean()


def collate_train(batch):
    xs, ys, fnames = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(fnames)


def collate_test(batch):
    xs, fnames = zip(*batch)
    return torch.stack(xs), list(fnames)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[index], lam * y + (1.0 - lam) * y[index]


def serializable_args(args: argparse.Namespace) -> dict:
    clean = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        if isinstance(value, Path):
            clean[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            clean[key] = value
    return clean


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for x, y, _ in tqdm(loader, desc="valid", unit="batch", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(float(loss.item()))
        ys.append(y.cpu().numpy())
        preds.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    lwlrap = label_ranking_average_precision_score(y_true, y_pred)
    return float(np.mean(losses)), float(lwlrap)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    root = Path(args.root).resolve()
    device = pick_device(args.device)
    classes = load_vocabulary(root)
    rows = load_training_rows(root, include_noisy=args.include_noisy)
    if args.limit:
        rows = rows.head(args.limit).copy()

    train_rows, val_rows = train_test_split(rows, test_size=args.val_size, random_state=args.seed)
    train_ds = AudioTaggingDataset(
        train_rows, classes, args.clip_seconds, "train", n_mels=args.n_mels,
        spec_augment=not args.no_spec_augment,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        n_freq_masks=args.n_freq_masks,
        n_time_masks=args.n_time_masks,
    )
    val_ds = AudioTaggingDataset(val_rows, classes, args.clip_seconds, "valid", n_mels=args.n_mels)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_train,
    )

    model = DeepAudioCNN(len(classes), dropout=args.dropout).to(device)
    if args.focal_gamma > 0:
        criterion = FocalBCEWithLogitsLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(0, args.warmup_epochs)
    cos_epochs = max(1, args.epochs - warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cos_epochs)

    print(
        f"device={device}, train={len(train_ds)}, valid={len(val_ds)}, "
        f"classes={len(classes)}, clip={args.clip_seconds}s"
    )
    best_lwlrap = -1.0
    patience_counter = 0
    history = []
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / args.model_name

    for epoch in range(1, args.epochs + 1):
        # --- linear warmup ---
        if epoch <= warmup:
            lr_scale = epoch / warmup
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * lr_scale
        model.train()
        running = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
        for x, y, _ in progress:
            x = x.to(device)
            y = y.to(device)
            x, y = mixup_batch(x, y, args.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running.append(float(loss.item()))
            progress.set_postfix(loss=f"{np.mean(running):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        val_loss, val_lwlrap = evaluate(model, val_loader, criterion, device)
        if epoch > warmup:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "val_loss": val_loss,
            "val_lwlrap": val_lwlrap,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_lwlrap > best_lwlrap + args.min_delta:
            best_lwlrap = val_lwlrap
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "args": serializable_args(args),
                    "best_lwlrap": best_lwlrap,
                },
                model_path,
            )
            print(f"saved best model: {model_path}")
        else:
            patience_counter += 1
            print(f"no improvement: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    metrics = {
        "best_lwlrap": best_lwlrap,
        "history": history,
        "model_path": str(model_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    (output_dir / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    device = pick_device(args.device)
    checkpoint = torch.load(root / "models" / args.model_name, map_location=device)
    classes = checkpoint["classes"]
    train_args = checkpoint.get("args", {})
    n_mels = int(train_args.get("n_mels", args.n_mels))
    clip_seconds = float(train_args.get("clip_seconds", args.clip_seconds))

    test_rows = pd.read_csv(root / "dataset" / "sample_submission.csv")
    test_rows = test_rows[["fname"]].copy()
    test_rows["audio_dir"] = str(root / "dataset" / "test")
    test_rows = rows_with_existing_audio(test_rows, root / "dataset" / "test")
    if args.limit:
        test_rows = test_rows.head(args.limit).copy()

    ds = AudioTaggingDataset(
        test_rows,
        classes,
        clip_seconds,
        "test",
        n_mels=n_mels,
        tta_crops=args.tta_crops,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_test,
    )
    model = DeepAudioCNN(len(classes), dropout=float(train_args.get("dropout", 0.25))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_fnames: list[str] = []
    all_scores: list[np.ndarray] = []
    for x, fnames in tqdm(loader, desc="predict", unit="batch"):
        if x.ndim == 5:
            batch, crops, channels, mels, frames = x.shape
            x = x.view(batch * crops, channels, mels, frames).to(device)
            scores = torch.sigmoid(model(x)).view(batch, crops, -1).mean(dim=1)
        else:
            scores = torch.sigmoid(model(x.to(device)))
        all_fnames.extend(fnames)
        all_scores.append(scores.cpu().numpy())

    scores_np = np.concatenate(all_scores, axis=0)
    sub = pd.DataFrame(scores_np, columns=classes)
    sub.insert(0, "fname", all_fnames)
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / args.output
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(sub)} rows, {len(classes)} labels)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FAT2019 PyTorch CNN baseline")
    parser.add_argument("--root", default=str(ROOT), help="Dataset/project root")
    parser.add_argument("--model-name", default="fat2019_cnn.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=128)

    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--include-noisy", action="store_true")
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--val-size", type=float, default=0.2)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--dropout", type=float, default=0.25)
    train_p.add_argument("--mixup-alpha", type=float, default=0.2)
    train_p.add_argument("--patience", type=int, default=10, help="Early stop patience")
    train_p.add_argument("--min-delta", type=float, default=1e-4, help="Min improvement to count as better")
    train_p.add_argument("--no-spec-augment", action="store_true", help="Disable SpecAugment")
    train_p.add_argument("--freq-mask-param", type=int, default=12, help="Max freq bins to mask")
    train_p.add_argument("--time-mask-param", type=int, default=24, help="Max time frames to mask")
    train_p.add_argument("--n-freq-masks", type=int, default=2, help="Number of freq masks")
    train_p.add_argument("--n-time-masks", type=int, default=2, help="Number of time masks")
    train_p.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (0 = plain BCE)")
    train_p.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha")
    train_p.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs")
    train_p.set_defaults(func=train)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--output", default="submission_cnn.csv")
    pred_p.add_argument("--tta-crops", type=int, default=3)
    pred_p.set_defaults(func=predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

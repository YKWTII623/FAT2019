#!/usr/bin/env python3
"""Curated-only baseline for Freesound Audio Tagging 2019.

The script intentionally uses only numpy/scipy/sklearn so it can run in a
minimal Python environment. It builds fixed-size log-mel summary features and
trains a multi-label one-vs-rest classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
BAD_CURATED_FILES = {
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
    "1d44b0bd.wav",
}


def load_vocabulary(root: Path) -> list[str]:
    vocab = pd.read_csv(root / "meta" / "vocabulary.csv", header=None, names=["idx", "label"])
    return vocab.sort_values("idx")["label"].tolist()


def split_labels(value: str) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


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


def logmel_summary(
    path: Path,
    target_sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
) -> np.ndarray:
    audio = read_wav_mono(path, target_sr)
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))

    _, _, zxx = stft(
        audio,
        fs=target_sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx).astype(np.float32) ** 2
    fb = mel_filterbank(target_sr, n_fft, n_mels, 20.0, target_sr / 2.0)
    mel = np.maximum(fb @ power, 1e-10)
    logmel = np.log(mel)

    stats = [
        logmel.mean(axis=1),
        logmel.std(axis=1),
        logmel.max(axis=1),
        logmel.min(axis=1),
        np.percentile(logmel, 25, axis=1),
        np.percentile(logmel, 75, axis=1),
    ]
    duration = np.array([audio.size / target_sr], dtype=np.float32)
    return np.concatenate(stats + [duration]).astype(np.float32)


def rows_with_existing_audio(df: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    exists = df["fname"].map(lambda name: (audio_dir / name).exists())
    return df.loc[exists].copy()


def load_training_rows(root: Path, include_noisy: bool) -> pd.DataFrame:
    curated = pd.read_csv(root / "meta" / "train_curated_post_competition.csv")
    curated = curated[~curated["fname"].isin(BAD_CURATED_FILES)].copy()
    curated["audio_dir"] = str(root / "dataset" / "train_curated")
    curated = rows_with_existing_audio(curated, root / "dataset" / "train_curated")
    frames = [curated]

    noisy_dir = root / "dataset" / "train_noisy"
    noisy_meta = root / "meta" / "train_noisy_post_competition.csv"
    if include_noisy and noisy_dir.exists() and noisy_meta.exists():
        noisy = pd.read_csv(noisy_meta)
        noisy["audio_dir"] = str(noisy_dir)
        noisy = rows_with_existing_audio(noisy, noisy_dir)
        frames.append(noisy)

    train = pd.concat(frames, ignore_index=True)
    if train.empty:
        raise RuntimeError("No training audio found. Check dataset/train_curated/ and metadata files.")
    return train


def feature_cache_name(prefix: str, rows: pd.DataFrame, limit: int | None) -> str:
    suffix = f"limit{limit}" if limit else f"n{len(rows)}"
    return f"{prefix}_{suffix}_logmel_stats.npz"


def extract_features(
    rows: pd.DataFrame,
    cache_path: Path,
    limit: int | None = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, list[str]]:
    if cache_path.exists() and not overwrite:
        print(f"load cached features: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["x"].astype(np.float32), data["fnames"].tolist()

    work = rows.head(limit).copy() if limit else rows.copy()
    features: list[np.ndarray] = []
    fnames: list[str] = []
    iterator = tqdm(
        work.itertuples(index=False),
        total=len(work),
        desc=f"extract {cache_path.stem}",
        unit="file",
    )
    for row in iterator:
        path = Path(row.audio_dir) / row.fname
        try:
            features.append(logmel_summary(path))
            fnames.append(row.fname)
        except Exception as exc:
            tqdm.write(f"[warn] skip {path}: {exc}")

    x = np.vstack(features).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, fnames=np.array(fnames, dtype=object))
    return x, fnames


def save_training_plots(
    output_dir: Path,
    classes: list[str],
    y: np.ndarray,
    val_scores: np.ndarray | None = None,
) -> list[str]:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skip plots: {exc}")
        return []

    output_dir.mkdir(exist_ok=True)
    saved: list[str] = []

    label_counts = y.sum(axis=0)
    order = np.argsort(label_counts)[::-1]
    fig_height = max(8, len(classes) * 0.18)
    plt.figure(figsize=(12, fig_height))
    plt.barh(np.array(classes)[order][::-1], label_counts[order][::-1], color="#2f7ed8")
    plt.xlabel("clips")
    plt.title("Training label distribution")
    plt.tight_layout()
    path = output_dir / "label_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved.append(str(path))

    if val_scores is not None:
        avg_scores = val_scores.mean(axis=0)
        top = np.argsort(avg_scores)[-25:]
        plt.figure(figsize=(10, 7))
        plt.barh(np.array(classes)[top], avg_scores[top], color="#34a853")
        plt.xlabel("mean predicted probability")
        plt.title("Top validation prediction scores")
        plt.tight_layout()
        path = output_dir / "validation_top_scores.png"
        plt.savefig(path, dpi=160)
        plt.close()
        saved.append(str(path))

    return saved


def make_targets(rows: pd.DataFrame, fnames: Iterable[str], classes: list[str]) -> np.ndarray:
    label_map = dict(zip(rows["fname"], rows["labels"]))
    mlb = MultiLabelBinarizer(classes=classes)
    target_labels = [split_labels(label_map[fname]) for fname in fnames]
    return mlb.fit_transform(target_labels).astype(np.int8)


def train(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    classes = load_vocabulary(root)
    train_rows = load_training_rows(root, include_noisy=args.include_noisy)
    cache = root / "features" / feature_cache_name("train", train_rows, args.limit)
    x, fnames = extract_features(train_rows, cache, limit=args.limit, overwrite=args.overwrite_features)
    y = make_targets(train_rows, fnames, classes)
    print(f"training rows: {len(fnames)}, labels: {len(classes)}, features: {x.shape[1]}")

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.val_size, random_state=args.seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    clf = OneVsRestClassifier(
        LogisticRegression(
            C=args.c,
            class_weight="balanced",
            max_iter=args.max_iter,
            solver="liblinear",
            random_state=args.seed,
        ),
        n_jobs=args.n_jobs,
    )
    print("fit One-vs-Rest Logistic Regression...")
    clf.fit(x_train, y_train)
    print("score validation split...")
    val_scores = clf.predict_proba(x_val)
    lwlrap = label_ranking_average_precision_score(y_val, val_scores)

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / args.model_name
    joblib.dump({"scaler": scaler, "model": clf, "classes": classes}, model_path)

    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    plot_paths = [] if args.no_plots else save_training_plots(output_dir, classes, y, val_scores)
    metrics = {
        "train_rows": int(len(train_rows)),
        "feature_rows": int(len(fnames)),
        "classes": len(classes),
        "val_size": args.val_size,
        "lwlrap": float(lwlrap),
        "model_path": str(model_path),
        "plots": plot_paths,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def predict(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    bundle = joblib.load(root / "models" / args.model_name)
    test_rows = pd.read_csv(root / "meta" / "test_post_competition.csv")
    test_rows["audio_dir"] = str(root / "dataset" / "test")
    test_rows = rows_with_existing_audio(test_rows, root / "dataset" / "test")
    cache = root / "features" / feature_cache_name("test", test_rows, args.limit)
    x, fnames = extract_features(test_rows, cache, limit=args.limit, overwrite=args.overwrite_features)
    print(f"predict rows: {len(fnames)}, features: {x.shape[1]}")
    x = bundle["scaler"].transform(x)
    scores = bundle["model"].predict_proba(x)

    sub = pd.DataFrame(scores, columns=bundle["classes"])
    sub.insert(0, "fname", fnames)
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / args.output
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(sub)} rows, {len(bundle['classes'])} labels)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FAT2019 curated-only sklearn baseline")
    parser.add_argument("--root", default=str(ROOT), help="Dataset/project root")
    parser.add_argument("--model-name", default="fat2019_logmel_logreg.joblib")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib training plots")

    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--include-noisy", action="store_true", help="Use audio_train_noisy if present")
    train_p.add_argument("--val-size", type=float, default=0.2)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--c", type=float, default=1.0)
    train_p.add_argument("--max-iter", type=int, default=1000)
    train_p.add_argument("--n-jobs", type=int, default=1)
    train_p.set_defaults(func=train)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--output", default="submission.csv")
    pred_p.set_defaults(func=predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# Freesound Audio Tagging 2019

参加 Kaggle Freesound Audio Tagging 2019 比赛的实验记录。任务是对 10 秒以内的音频片段打 80 个环境声音标签，每个片段可以同时属于多个类别。

目前验证集 LWLRAP 做到了 **0.620**（比赛 top 1 是 0.755）。

## 数据

- curated: 4,970 条，人工标注，标签比较靠谱
- noisy: 19,815 条，从 Flickr 视频自动标注的，不太靠谱
- test: 3,361 条

80 个类别长尾分布很严重，有些类只有几十个样本，有些几千个。

## 思路

特征用的是 log-mel spectrogram（128 bins，5 秒随机裁剪），模型是自己搭的一个 18 层的 CNN，大概 1500 万参数。

主要加了这些东西：

- **SpecAugment**：在 spectrogram 上随机盖掉一些频率和时间段
- **Focal Loss**：让模型多关注难分和稀有的类别
- **Warmup + Cosine Annealing**：前 5 个 epoch 慢慢加大学习率，后面余弦退火
- **Mixup**：训练时随机混合两个样本
- **早停**：10 个 epoch 验证集不涨就停

详细代码在 `src/fat2019_cnn.py`。

## 跑起来

```bash
# 装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scipy scikit-learn joblib tqdm matplotlib

# 训练
python src/fat2019_cnn.py train --include-noisy --batch-size 64

# 预测
python src/fat2019_cnn.py predict
```

输出在 `outputs/submission_cnn.csv`。

## 项目结构

```
src/          — 代码（sklearn 基线 + PyTorch CNN）
notebooks/    — Kaggle 提交用的 notebook
meta/         — 标签和文件列表
reports/      — 实验报告
```

## 参考

- 比赛页面：https://www.kaggle.com/c/freesound-audio-tagging-2019
- top 方案：https://github.com/ebouteillon/freesound-audio-tagging-2019

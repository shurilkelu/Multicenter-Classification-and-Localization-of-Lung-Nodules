# LUNA16 轻量 3D CNN + 小型 Transformer Encoder 结节判定与定位实验说明

本文档记录本次 LUNA16 肺结节检测实验的完整研究内容，包括研究目标、数据集划分、模型设计、训练加速策略、运行方式、输出文件、评估指标、预测结果 CSV、结节可视化图片以及后续改进建议。

本项目主代码文件为：

```text
train_luna16_3d_cnn_transformer.py
```

当前工作目录为：

```text
/Users/wangshang/Documents/New project
```

LUNA16 数据集目录为：

```text
/Users/wangshang/LUNA16
```

最终输出目录为：

```text
luna16_3d_outputs/
```

## 1. 研究目标

本次实验目标是基于 LUNA16 CT 数据集，设计并训练一个较轻量的 3D 深度学习模型，用于完成两个任务：

1. 判断一个 3D CT patch 是否包含肺结节。
2. 对真实结节样本预测结节中心位置，也就是输出结节相对于候选 patch 中心的三维偏移。

模型采用多任务学习结构：

```text
输入 3D CT patch
  -> 轻量 3D CNN 提取局部空间特征
  -> 小型 Transformer Encoder 建模 patch 内部全局上下文
  -> 分类头输出是否为结节
  -> 定位头输出结节中心偏移 dx, dy, dz
```

本实验不是完整的临床级结节检测系统，而是一个基于候选结节 patch 的二分类与局部定位模型。它的输入已经是围绕候选点裁剪出的 3D patch，因此模型主要解决：

```text
候选点是否真的是结节？
如果是，结节中心在哪里？
```

## 2. 数据集说明

使用的数据集为 LUNA16。LUNA16 是肺结节检测领域常用公开数据集，包含多个 subset，每个 subset 下是 CT 体数据文件。

本项目使用的主要文件包括：

```text
/Users/wangshang/LUNA16/annotations.csv
/Users/wangshang/LUNA16/subset0/
/Users/wangshang/LUNA16/subset1/
/Users/wangshang/LUNA16/subset2/
/Users/wangshang/LUNA16/subset3/
/Users/wangshang/LUNA16/subset4/
```

每个 subset 中包含 `.mhd` 和 `.raw` 文件。其中：

- `.mhd` 保存 CT 图像的元信息，例如 spacing、origin、size。
- `.raw` 保存实际体素数据。
- `annotations.csv` 保存真实结节中心坐标和直径。

代码使用 SimpleITK 读取 `.mhd`，再自动关联对应 `.raw`。

## 3. 本次实验的数据划分

根据本次最终要求，数据划分如下：

| 数据用途 | subset | CT 数量 | 阳性结节样本 | 阴性样本 | 总 patch 数 |
|---|---:|---:|---:|---:|---:|
| 训练集 | subset0, subset1, subset2 | 267 | 368 | 184 | 552 |
| 验证集 | subset3 | 89 | 119 | 60 | 179 |
| 测试集 | subset4 | 89 | 128 | 64 | 192 |

说明：

- `subset0, subset1, subset2` 用于训练。
- `subset3` 用于验证和选择最佳 checkpoint。
- `subset4` 用于最终测试。
- `subset5` 当前没有参与本次最终实验。
- 阴性样本由代码随机采样生成，并要求距离已知结节至少 `40 mm`，避免阴性点过于靠近真实结节。

对应代码位置：

```python
TRAIN_SUBSETS = (0, 1, 2)
VAL_SUBSETS = (3,)
TEST_SUBSETS = (4,)
```

## 4. 样本构建方式

每个样本被封装为一个 `PatchSample`，其中包含：

| 字段 | 含义 |
|---|---|
| `seriesuid` | CT 序列 ID |
| `mhd_path` | 对应 CT 的 `.mhd` 路径 |
| `label` | 是否为结节，1 为结节，0 为非结节 |
| `subset` | 所属 subset 编号 |
| `candidate_center_xyz` | 候选点中心，世界坐标系，单位 mm |
| `nodule_center_xyz` | 真实结节中心，阴性样本为 None |
| `diameter_mm` | 真实结节直径，阴性样本为 0 |

阳性样本来自 `annotations.csv` 中的真实结节标注。

阴性样本通过如下流程生成：

1. 在指定 subset 的 CT 体数据中随机选取体素位置。
2. 转换为世界坐标。
3. 计算该点到当前 CT 内所有真实结节中心的距离。
4. 若距离所有真实结节都大于 `NEG_MIN_DIST_MM = 40.0`，则作为阴性样本。

这样可以减少把真实结节附近区域错误当作阴性的风险。

## 5. CT patch 裁剪和归一化

本次最终超快版本使用的 patch 大小为：

```text
24 x 24 x 24
```

命令参数为：

```bash
--patch-size 24 24 24
```

每个 patch 是以候选点为中心，从原始 CT 中裁剪出的三维局部区域。

灰度值处理方式：

1. 将 CT HU 值截断到：

```text
HU_MIN = -1000
HU_MAX = 400
```

2. 线性归一化到 `[0, 1]`。

3. 再映射到 `[-1, 1]`。

代码逻辑为：

```python
patch = np.clip(patch, HU_MIN, HU_MAX)
patch = (patch - HU_MIN) / (HU_MAX - HU_MIN)
patch = patch * 2.0 - 1.0
```

这样做的目的：

- 去掉肺窗之外的极端 HU 值。
- 让神经网络输入数值范围更稳定。
- 加快训练收敛。

## 6. 定位标签定义

定位任务不是直接回归世界坐标，而是回归结节中心相对于候选 patch 中心的归一化偏移。

模型输出：

```text
pred_offset_x, pred_offset_y, pred_offset_z
```

每个 offset 被限制在 `[-1, 1]` 之间。

真实偏移计算方式：

```text
target_offset_xyz = (target_idx_xyz - center_idx_xyz) / (patch_size_xyz / 2)
```

其中：

- `target_idx_xyz` 是真实结节中心对应的体素坐标。
- `center_idx_xyz` 是候选 patch 中心对应的体素坐标。
- `patch_size_xyz / 2` 用于归一化。

评估时再将预测偏移转换回世界坐标：

```text
pred_center_xyz = candidate_center_xyz
                  + pred_offset_xyz * (patch_size_xyz / 2) * spacing_xyz
```

因此最终 CSV 中同时保存：

- 候选点坐标。
- 真实结节中心坐标。
- 预测结节中心坐标。
- 真实偏移。
- 预测偏移。
- 定位误差，单位 mm。

## 7. 模型结构

本次最终使用的是轻量 3D CNN + 小型 Transformer Encoder。

模型类名：

```python
Light3DCNNTransformer
```

### 7.1 输入

输入张量形状：

```text
[batch_size, channels, depth, height, width]
```

本实验中为：

```text
[B, 1, 24, 24, 24]
```

### 7.2 轻量 3D CNN 主干

CNN 结构如下：

```text
ConvBlock3D(1 -> base_channels, stride=1)
ConvBlock3D(base_channels -> base_channels*2, stride=2)
DepthwiseSeparableConv3D(base_channels*2 -> base_channels*4, stride=2)
DepthwiseSeparableConv3D(base_channels*4 -> embed_dim, stride=2)
```

本次参数为：

```text
base_channels = 4
embed_dim = 32
```

因此通道变化大致为：

```text
1 -> 4 -> 8 -> 16 -> 32
```

空间尺寸变化：

```text
24 x 24 x 24
  -> 12 x 12 x 12
  -> 6 x 6 x 6
  -> 3 x 3 x 3
```

最终得到 `3 x 3 x 3 = 27` 个 token，每个 token 的维度为 `32`。

### 7.3 深度可分离 3D 卷积

为了加快训练，本模型使用 `DepthwiseSeparableConv3D` 替代一部分普通 3D 卷积。

普通 3D 卷积计算量较大，而深度可分离 3D 卷积分为两步：

1. depthwise 3D 卷积：每个通道独立卷积。
2. pointwise 1x1x1 卷积：融合通道信息。

这样可以明显减少参数量和计算量。

### 7.4 Transformer Encoder

CNN 输出的 3D feature map 被展平成 token 序列：

```text
[B, C, D, H, W] -> [B, D*H*W, C]
```

本实验中：

```text
[B, 32, 3, 3, 3] -> [B, 27, 32]
```

Transformer 配置：

| 参数 | 数值 |
|---|---:|
| `embed_dim` | 32 |
| `transformer_layers` | 1 |
| `transformer_heads` | 2 |
| `dim_feedforward` | 64 |
| `dropout` | 0.1 |

Transformer 的作用是让 patch 内不同空间位置之间建立全局关系，帮助模型理解结节和周围结构的上下文。

### 7.5 输出头

模型有两个输出头：

1. 分类头 `cls_head`

```text
Linear(embed_dim -> embed_dim/2)
GELU
Dropout
Linear(embed_dim/2 -> 1)
```

输出一个 logit，经过 sigmoid 后得到：

```text
prob_nodule
```

2. 定位头 `loc_head`

```text
Linear(embed_dim -> embed_dim/2)
GELU
Dropout
Linear(embed_dim/2 -> 3)
Tanh
```

输出：

```text
pred_offset_x, pred_offset_y, pred_offset_z
```

由于最后使用 `Tanh`，定位偏移被限制在 `[-1, 1]`。

## 8. 损失函数

本项目使用多任务损失：

```text
total_loss = cls_loss + loc_weight * loc_loss
```

### 8.1 分类损失

分类损失使用：

```python
nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

其中 `pos_weight` 根据训练集正负样本比例自动计算。

本实验训练集为：

```text
positive = 368
negative = 184
```

因为阳性多于阴性，所以 `pos_weight` 最小取 `1.0`。

### 8.2 定位损失

定位损失使用：

```python
SmoothL1Loss
```

只对阳性样本计算定位损失。阴性样本没有真实结节中心，因此不参与定位损失。

通过 `loc_mask` 控制：

```text
loc_mask = 1 表示阳性样本，参与定位损失
loc_mask = 0 表示阴性样本，不参与定位损失
```

本实验最终参数：

```text
loc_weight = 1.0
```

## 9. 训练加速策略

最初版本单个 epoch 可能需要数分钟。为了在紧急时间内完成 80 epoch，本次进行了多项加速。

### 9.1 使用更小 patch

从较大的 3D patch 改为：

```text
24 x 24 x 24
```

好处：

- 输入体素数量更少。
- 3D 卷积计算量显著下降。
- Transformer token 数减少到 27。

代价：

- 上下文范围变小。
- 对大结节或边界信息的表达可能变弱。

### 9.2 使用更轻量模型

最终模型参数：

```text
base_channels = 4
embed_dim = 32
transformer_layers = 1
transformer_heads = 2
batch_size = 64
```

相较之前的更大模型，计算量大幅降低。

### 9.3 不进行 1 mm 重采样

最终默认使用：

```bash
--no-resample
```

也就是说直接使用原始 CT spacing 裁剪 patch，不把 CT 全部重采样到 1 mm 各向同性。

好处：

- 避免读取 CT 后做大规模重采样。
- 预缓存速度更快。

代价：

- 不同 CT 的体素间距不完全一致。
- 物理尺度一致性不如重采样版本。

### 9.4 预缓存 patch 到磁盘

最终默认开启：

```bash
--precache-patches
--cache-float16
```

训练前先把所有 3D patch 裁剪出来，并保存到：

```text
luna16_3d_outputs/patch_cache_ultrafast/
```

这样每个 epoch 不再反复读取整套 CT 体数据，只读取已经裁剪好的小 patch。

### 9.5 将缓存 patch 读入内存

最终默认开启：

```bash
--cache-in-memory
```

这会在训练前把 `.npz` patch 加载到 RAM 中。训练时直接从内存取 patch，进一步减少磁盘 I/O。

### 9.6 减少负样本比例

本实验使用：

```bash
--negative-ratio 0.5
```

也就是负样本数量约为阳性样本数量的一半。这样可以减少训练样本总数，提高速度。

### 9.7 最终训练速度

最终超快版本在缓存完成后，每个 epoch 大约数秒即可完成。

完整 80 epoch 成功跑完，并生成测试集结果和图片可视化。

## 10. 运行环境

使用的 Python 解释器：

```text
/Applications/anaconda3/bin/python3
```

主要依赖：

```text
torch
SimpleITK
numpy
pandas
matplotlib
scikit-learn
tqdm
```

依赖文件：

```text
requirements_3d.txt
```

当前设备自动选择逻辑：

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

本次训练实际使用：

```text
mps
```

也就是 Apple Silicon 的 Metal 后端。

## 11. 运行命令

### 11.1 一键训练 80 epoch

可以运行：

```bash
./run_80epoch_fast_training.sh
```

该脚本内部调用：

```bash
./run_fast_training.sh
```

当前 `run_fast_training.sh` 使用的主要参数为：

```bash
/Applications/anaconda3/bin/python3 train_luna16_3d_cnn_transformer.py \
  --train-subsets 0 1 2 \
  --val-subsets 3 \
  --test-subsets 4 \
  --precache-patches \
  --cache-in-memory \
  --cache-float16 \
  --no-resample \
  --negative-ratio 0.5 \
  --patch-size 24 24 24 \
  --batch-size 64 \
  --base-channels 4 \
  --embed-dim 32 \
  --transformer-layers 1 \
  --transformer-heads 2 \
  --dropout 0.1 \
  --lr 5e-4 \
  --loc-weight 1.0 \
  --num-workers 0 \
  --epochs 80 \
  --patience 1000 \
  --disable-early-stopping \
  --positive-jitter-mm 2.0
```

### 11.2 只评估，不重新训练

如果已经有 checkpoint，可以只跑测试集评估和输出图片：

```bash
/Applications/anaconda3/bin/python3 train_luna16_3d_cnn_transformer.py --eval-only
```

该命令会读取：

```text
luna16_3d_outputs/best_light3d_cnn_transformer_luna16.pth
```

并重新生成：

```text
test_predictions_3d.csv
test_nodule_predictions_3d.csv
test_metrics_3d.json
test_nodule_visualizations/
```

### 11.3 VS Code 中运行

项目中已经包含：

```text
.vscode/launch.json
LUNA16-3D-CNN-Transformer.code-workspace
```

在 VS Code 中可以选择启动项：

```text
Ultra Fast Train LUNA16 3D CNN Transformer
```

它使用与 `run_fast_training.sh` 一致的超快训练参数。

## 12. 输出文件说明

所有主要输出都在：

```text
luna16_3d_outputs/
```

### 12.1 模型文件

```text
luna16_3d_outputs/best_light3d_cnn_transformer_luna16.pth
```

这是验证集 AUC 最优时保存的模型 checkpoint。

本次最优 epoch：

```text
epoch = 29
```

最优验证集结果：

| 指标 | 数值 |
|---|---:|
| Best val AUC | 0.9548 |
| Best val accuracy | 0.8771 |
| Best val F1 | 0.9076 |
| Best val mean location error | 0.7429 mm |

### 12.2 训练历史

```text
luna16_3d_outputs/training_history_3d.csv
```

包含 80 行，每行对应一个 epoch。

主要列包括：

| 列名 | 含义 |
|---|---|
| `epoch` | epoch 编号 |
| `lr` | 当前学习率 |
| `train_loss` | 训练总损失 |
| `train_accuracy` | 训练准确率 |
| `train_auc` | 训练 AUC |
| `train_f1` | 训练 F1 |
| `val_loss` | 验证总损失 |
| `val_accuracy` | 验证准确率 |
| `val_auc` | 验证 AUC |
| `val_f1` | 验证 F1 |
| `val_loc_error_mean_mm` | 验证集平均定位误差 |
| `val_loc_error_median_mm` | 验证集中位定位误差 |

### 12.3 测试指标

```text
luna16_3d_outputs/test_metrics_3d.json
```

本次最终测试集结果：

| 指标 | 数值 |
|---|---:|
| Loss | 0.2760 |
| Classification loss | 0.2754 |
| Location loss | 0.0006 |
| Accuracy | 0.9115 |
| Precision | 0.9111 |
| Recall | 0.9609 |
| F1 | 0.9354 |
| AUC | 0.9405 |
| Mean location error | 0.6512 mm |
| Median location error | 0.4761 mm |
| Saved prediction images | 128 |

### 12.4 全量测试预测结果

```text
luna16_3d_outputs/test_predictions_3d.csv
```

该文件包含 subset4 测试集全部 192 个样本，包括阳性和阴性。

主要列：

| 列名 | 含义 |
|---|---|
| `seriesuid` | CT 序列 ID |
| `label` | 真实标签，1 为结节，0 为非结节 |
| `prob_nodule` | 模型预测为结节的概率 |
| `pred_label` | 按阈值 0.5 得到的预测标签 |
| `diameter_mm` | 真实结节直径，阴性样本为 0 |
| `candidate_x/y/z` | 候选点中心坐标 |
| `target_x/y/z` | 真实结节中心坐标 |
| `pred_x/y/z` | 模型预测结节中心坐标 |
| `target_offset_x/y/z` | 真实归一化偏移 |
| `pred_offset_x/y/z` | 模型预测归一化偏移 |
| `loc_error_mm` | 定位误差，单位 mm |
| `visualization_path` | 对应三视图结节可视化图片路径 |

阴性样本没有真实结节中心，因此 `loc_error_mm` 为 NaN。

### 12.5 仅结节预测结果

```text
luna16_3d_outputs/test_nodule_predictions_3d.csv
```

这是从 `test_predictions_3d.csv` 中筛出的真实结节样本，共 128 行。

它更适合查看：

- 模型对每个真实结节的判定概率。
- 是否漏判。
- 预测结节中心坐标。
- 真实结节中心坐标。
- 定位误差。
- 对应三视图图片路径。

本次真实结节样本统计：

| 项目 | 数值 |
|---|---:|
| subset4 真实结节数 | 128 |
| 正确判定为结节 | 123 |
| 漏判为非结节 | 5 |

### 12.6 结节三视图可视化图片

```text
luna16_3d_outputs/test_nodule_visualizations/
```

本次共保存：

```text
128 张 PNG
```

每张图片包含三个视图：

| 视图 | 含义 |
|---|---|
| Axial | 横断面 |
| Coronal | 冠状面 |
| Sagittal | 矢状面 |

图中标记：

| 标记 | 含义 |
|---|---|
| 绿色 `x` | 真实结节中心 |
| 红色 `+` | 模型预测结节中心 |

图片标题包含：

```text
seriesuid
prob_nodule
pred label
diameter_mm
loc_error_mm
```

示例图片：

![准确定位示例](luna16_3d_outputs/test_nodule_visualizations/nodule_0006_1_3_6_1_4_1_14519_5_2_1_6279_6001_107351566259572521472765997306.png)

该样本预测结果：

```text
prob = 0.971
pred = Nodule
diameter = 4.21 mm
loc_error = 0.03 mm
```

漏判示例：

![漏判示例](luna16_3d_outputs/test_nodule_visualizations/nodule_0037_1_3_6_1_4_1_14519_5_2_1_6279_6001_161855583909753609742728521805.png)

该样本预测结果：

```text
prob = 0.095
pred = Non-nodule
diameter = 7.43 mm
loc_error = 1.91 mm
```

## 13. 训练曲线和评估图

### 13.1 损失曲线

```text
luna16_3d_outputs/loss_curve_3d.png
```

![Loss curve](luna16_3d_outputs/loss_curve_3d.png)

该图展示训练集和验证集总损失随 epoch 的变化。

### 13.2 准确率曲线

```text
luna16_3d_outputs/accuracy_curve_3d.png
```

![Accuracy curve](luna16_3d_outputs/accuracy_curve_3d.png)

该图展示训练准确率和验证准确率变化。

### 13.3 AUC 曲线

```text
luna16_3d_outputs/auc_curve_3d.png
```

![AUC curve](luna16_3d_outputs/auc_curve_3d.png)

该图展示训练 AUC 和验证 AUC 变化。

### 13.4 定位误差曲线

```text
luna16_3d_outputs/location_error_curve_3d.png
```

![Location error curve](luna16_3d_outputs/location_error_curve_3d.png)

该图展示验证集平均定位误差随 epoch 的变化。

### 13.5 混淆矩阵

```text
luna16_3d_outputs/confusion_matrix_3d.png
```

![Confusion matrix](luna16_3d_outputs/confusion_matrix_3d.png)

本次测试集混淆矩阵：

| 真实类别 / 预测类别 | Pred Negative | Pred Positive |
|---|---:|---:|
| True Negative | 52 | 12 |
| True Positive | 5 | 123 |

解释：

- 52 个阴性样本被正确判定为阴性。
- 12 个阴性样本被误判为结节。
- 5 个真实结节被漏判。
- 123 个真实结节被正确判定为结节。

### 13.6 ROC 曲线

```text
luna16_3d_outputs/roc_curve_3d.png
```

![ROC curve](luna16_3d_outputs/roc_curve_3d.png)

测试集 AUC：

```text
0.9405
```

### 13.7 PR 曲线

```text
luna16_3d_outputs/pr_curve_3d.png
```

![PR curve](luna16_3d_outputs/pr_curve_3d.png)

PR 曲线展示 precision 和 recall 之间的关系。对于结节检测任务，recall 很重要，因为漏判结节通常比多报一些候选更严重。

## 14. 本次结果解读

### 14.1 分类性能

本次测试集结果：

```text
Accuracy = 0.9115
Precision = 0.9111
Recall = 0.9609
F1 = 0.9354
AUC = 0.9405
```

从这些指标看，模型在 subset4 上有较好的结节识别能力。

其中 recall 为 `0.9609`，表示真实结节中绝大多数都被识别出来：

```text
123 / 128
```

漏判数量：

```text
5
```

precision 为 `0.9111`，说明模型判定为结节的样本中，大部分确实是结节，但仍有一定假阳性。

### 14.2 定位性能

定位误差：

```text
Mean location error = 0.6512 mm
Median location error = 0.4761 mm
```

这说明在被标注为真实结节的 patch 中，模型预测中心与真实中心通常非常接近。

误差最小样本之一：

```text
loc_error = 0.0271 mm
```

误差较大样本之一：

```text
loc_error = 3.6317 mm
```

需要注意，当前定位是在已裁剪的候选 patch 内进行局部回归，不是对整张 CT 做全局结节搜索。

### 14.3 为什么定位误差看起来较低

本任务的输入 patch 是以候选点为中心裁剪的，而阳性样本候选点来自真实标注中心或带少量 jitter 的中心。因此定位任务相对整图检测更简单。

换句话说：

```text
模型不是从整张 CT 中从零寻找所有结节，
而是在候选 patch 中判断和微调结节位置。
```

因此定位误差较低是合理的，但不能直接等同于完整肺结节检测系统在整张 CT 上的定位误差。

## 15. 代码结构说明

核心代码都在：

```text
train_luna16_3d_cnn_transformer.py
```

主要模块如下。

### 15.1 数据结构

```python
VolumeInfo
PatchSample
VolumeData
```

用于保存 CT 元信息、patch 样本信息和体数据。

### 15.2 坐标转换

```python
world_to_index_xyz
index_to_world_xyz
```

用于在世界坐标和体素坐标之间转换。

LUNA16 标注使用世界坐标，CT 数组裁剪使用体素坐标，因此这两类函数很关键。

### 15.3 CT 读取

```python
load_volume
VolumeCache
```

负责读取 `.mhd` CT，并在必要时缓存体数据，避免重复读取。

### 15.4 patch 构建

```python
build_patch_item
crop_patch_zyx
normalize_patch_hu
augment_patch
```

负责：

- 根据候选中心裁剪 patch。
- 对 HU 值归一化。
- 对训练样本进行翻转、亮度扰动等轻量增强。
- 生成分类标签和定位标签。

### 15.5 Dataset

```python
Luna3DPatchDataset
CachedLuna3DPatchDataset
```

其中 `CachedLuna3DPatchDataset` 是本次加速的核心之一，用于读取预裁剪 patch，并可以将其加载到内存。

### 15.6 模型

```python
ConvBlock3D
DepthwiseSeparableConv3D
Light3DCNNTransformer
```

模型由轻量 3D CNN、小型 Transformer、分类头和定位头组成。

### 15.7 训练和评估

```python
train_one_epoch
evaluate
multitask_loss
summarize_epoch
```

`evaluate` 会完成：

- 分类指标统计。
- 定位误差统计。
- 输出预测 CSV。
- 输出结节三视图可视化图片。

### 15.8 可视化输出

```python
save_plots
save_nodule_prediction_image
```

`save_plots` 输出训练曲线、混淆矩阵、ROC、PR 曲线。

`save_nodule_prediction_image` 输出每个真实结节的三视图图像，并标注真实位置和预测位置。

## 16. 关键参数汇总

| 参数 | 数值 | 说明 |
|---|---:|---|
| `patch_size` | 24 24 24 | 输入 3D patch 大小 |
| `negative_ratio` | 0.5 | 负样本数量约为阳性一半 |
| `positive_jitter_mm` | 2.0 | 训练时阳性中心随机扰动 |
| `batch_size` | 64 | 批大小 |
| `epochs` | 80 | 训练轮数 |
| `lr` | 5e-4 | 学习率 |
| `weight_decay` | 1e-4 | AdamW 权重衰减 |
| `base_channels` | 4 | CNN 基础通道数 |
| `embed_dim` | 32 | Transformer token 维度 |
| `transformer_layers` | 1 | Transformer 层数 |
| `transformer_heads` | 2 | 注意力头数 |
| `dropout` | 0.1 | dropout |
| `loc_weight` | 1.0 | 定位损失权重 |
| `cache_float16` | True | patch 缓存使用 float16 |
| `cache_in_memory` | True | 训练前把 patch 加载进内存 |
| `no_resample` | True | 不进行 1 mm 重采样 |

## 17. 输出 CSV 示例解释

`test_nodule_predictions_3d.csv` 中一条预测较好的样本：

```text
seriesuid = 1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306
prob_nodule = 0.970727
pred_label = 1
diameter_mm = 4.214452
pred_x = -46.792828
pred_y = -66.980652
pred_z = -207.515427
target_x = -46.783730
target_y = -66.973694
target_z = -207.490891
loc_error_mm = 0.027078
```

解释：

- 模型认为该 patch 是结节的概率为 `0.970727`。
- `pred_label = 1`，表示判定为结节。
- 预测中心与真实中心距离约 `0.027 mm`。
- 这是一个定位非常准确的样本。

一条漏判样本：

```text
seriesuid = 1.3.6.1.4.1.14519.5.2.1.6279.6001.161855583909753609742728521805
prob_nodule = 0.094697
pred_label = 0
diameter_mm = 7.428431
loc_error_mm = 1.908435
```

解释：

- 该样本真实为结节。
- 但模型给出的结节概率只有 `0.094697`。
- 因为低于阈值 `0.5`，所以被判定为非结节。
- 这属于漏判。

## 18. 当前方法的优点

1. 训练速度快。

通过小 patch、轻量 CNN、小 Transformer、预缓存和内存缓存，80 epoch 可以在较短时间内完成。

2. 输出完整。

本项目输出了：

- loss 曲线。
- accuracy 曲线。
- AUC 曲线。
- 定位误差曲线。
- 混淆矩阵。
- ROC 曲线。
- PR 曲线。
- 全量测试预测 CSV。
- 仅结节预测 CSV。
- 结节三视图可视化图片。

3. 兼顾分类和定位。

模型不是只判断是否结节，还输出结节中心位置偏移和最终世界坐标。

4. 代码可直接复现实验。

训练、评估和可视化都集中在一个脚本中，VS Code 启动配置也已经准备好。

## 19. 当前方法的局限

1. 不是完整 CT 级检测系统。

当前模型基于候选 patch，不负责从整张 CT 中搜索所有候选结节。

2. patch 较小。

`24 x 24 x 24` patch 训练很快，但上下文较少。对边界模糊、大结节或周围结构复杂的样本可能不够稳。

3. 未做 1 mm 各向同性重采样。

这提高了速度，但不同 CT 的体素尺度可能存在差异。

4. 阴性样本数量较少。

当前 `negative_ratio = 0.5`，测试速度快，但真实临床筛查中阴性候选远多于阳性候选。

5. 测试集只使用 subset4。

当前结果只代表该划分下 subset4 的表现。如果换成更多 subset 或交叉验证，结果可能变化。

6. 可视化图片是 patch 层面的局部视图。

三视图图片展示的是模型输入 patch，而不是完整 CT 切片。

## 20. 后续改进建议

### 20.1 更完整的数据评估

可以继续扩展为：

```text
subset0-7 train
subset8 val
subset9 test
```

或者使用 10-fold cross validation，更符合 LUNA16 常见评估方式。

### 20.2 更真实的候选检测流程

当前候选点来自标注和随机负样本。后续可以接入候选生成算法，例如：

- 肺实质分割。
- 阈值和连通域候选。
- 多尺度候选筛选。
- 滑窗式 3D proposal。

这样可以从完整 CT 中生成候选，再交给本模型判别。

### 20.3 调整阈值

当前分类阈值为：

```text
0.5
```

如果目标是减少漏判，可以降低阈值，例如：

```text
0.3
```

这样 recall 可能进一步提高，但假阳性也会增加。

### 20.4 使用更大 patch 重新训练

如果时间允许，可以尝试：

```text
32 x 32 x 32
48 x 48 x 48
```

更大的 patch 能提供更多上下文，但训练速度会下降。

### 20.5 加强定位监督

可以尝试：

- 增大 `loc_weight`。
- 使用欧氏距离损失。
- 对大结节和小结节分组分析误差。
- 输出直径回归分支。

### 20.6 进一步输出报告

可以基于当前 CSV 自动生成：

- 最准确定位的前 20 个样本。
- 定位误差最大的前 20 个样本。
- 所有漏判结节图片。
- 所有假阳性候选图片。
- 按结节直径分组的 recall。

## 21. 文件索引

| 文件或目录 | 说明 |
|---|---|
| `train_luna16_3d_cnn_transformer.py` | 主训练、评估、可视化脚本 |
| `run_fast_training.sh` | 超快训练脚本 |
| `run_80epoch_fast_training.sh` | 一键 80 epoch 训练脚本 |
| `.vscode/launch.json` | VS Code 启动配置 |
| `requirements_3d.txt` | 依赖列表 |
| `luna16_3d_outputs/best_light3d_cnn_transformer_luna16.pth` | 最佳模型 checkpoint |
| `luna16_3d_outputs/training_history_3d.csv` | 训练历史 |
| `luna16_3d_outputs/test_metrics_3d.json` | 测试指标 |
| `luna16_3d_outputs/test_predictions_3d.csv` | 全量测试预测 |
| `luna16_3d_outputs/test_nodule_predictions_3d.csv` | 真实结节预测汇总 |
| `luna16_3d_outputs/test_nodule_visualizations/` | 真实结节三视图图片 |
| `luna16_3d_outputs/loss_curve_3d.png` | 损失曲线 |
| `luna16_3d_outputs/accuracy_curve_3d.png` | 准确率曲线 |
| `luna16_3d_outputs/auc_curve_3d.png` | AUC 曲线 |
| `luna16_3d_outputs/location_error_curve_3d.png` | 定位误差曲线 |
| `luna16_3d_outputs/confusion_matrix_3d.png` | 混淆矩阵 |
| `luna16_3d_outputs/roc_curve_3d.png` | ROC 曲线 |
| `luna16_3d_outputs/pr_curve_3d.png` | PR 曲线 |

## 22. 结论

本次实验完成了一个基于 LUNA16 的轻量化 3D CNN + 小型 Transformer Encoder 多任务模型。模型能够对候选 3D CT patch 判断是否为肺结节，并输出结节中心位置。

在最终划分：

```text
train = subset0, subset1, subset2
val = subset3
test = subset4
```

上，模型取得：

```text
Accuracy = 0.9115
F1 = 0.9354
AUC = 0.9405
Mean location error = 0.6512 mm
Median location error = 0.4761 mm
```

同时，项目已经输出完整的训练曲线、评估图、预测 CSV 和 128 张结节三视图可视化图片。整体上，本项目已经形成了一个可以训练、评估、可视化和复现实验结果的完整小型研究流程。

需要强调的是，该模型仍属于候选 patch 级别研究模型，不应直接用于临床诊断。如果要进一步发展成完整肺结节检测系统，还需要加入候选生成、全 CT 扫描、多尺度检测、更严格交叉验证和临床级误差分析。

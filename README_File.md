# LUNA16 Lightweight 3D CNN + Small Transformer Encoder Nodule Classification and Localization Experiment

This document records the complete research content of the LUNA16 pulmonary nodule experiment, including the research objective, dataset split, model design, training acceleration strategy, running method, output files, evaluation metrics, prediction CSV files, nodule visualization images, and suggestions for future improvement.

The main code file of this project is:

``` text
train_luna16_3d_cnn_transformer.py
```

The current working directory is:

``` text
/Users/wangshang/Documents/New project
```

The LUNA16 dataset directory is:

``` text
/Users/wangshang/LUNA16
```

The final output directory is:

``` text
luna16_3d_outputs/
```

## 1. Research Objective

The objective of this experiment is to design and train a lightweight 3D deep learning model based on the LUNA16 CT dataset. The model performs two tasks:

1.  Determine whether a 3D CT patch contains a pulmonary nodule.
2.  For true nodule samples, predict the nodule center location, namely the three-dimensional offset of the nodule center relative to the candidate patch center.

The model uses a multi-task learning structure:

``` text
Input 3D CT patch
  -> lightweight 3D CNN for local spatial feature extraction
  -> small Transformer Encoder for global contextual modeling inside the patch
  -> classification head outputs whether the patch is a nodule
  -> localization head outputs nodule center offsets dx, dy, dz
```

This experiment is not a complete clinical-level pulmonary nodule detection system. It is a candidate-patch-based binary classification and local localization model. Its input is already a 3D patch cropped around a candidate point, so the model mainly answers:

``` text
Is this candidate point a real nodule?
If yes, where is the nodule center?
```

## 2. Dataset Description

The dataset used in this project is LUNA16. LUNA16 is a commonly used public dataset for pulmonary nodule detection. It contains multiple subsets, and each subset contains CT volume files.

The main files used in this project are:

``` text
/Users/wangshang/LUNA16/annotations.csv
/Users/wangshang/LUNA16/subset0/
/Users/wangshang/LUNA16/subset1/
/Users/wangshang/LUNA16/subset2/
/Users/wangshang/LUNA16/subset3/
/Users/wangshang/LUNA16/subset4/
```

Each subset contains `.mhd` and `.raw` files:

-   `.mhd` stores the metadata of the CT image, such as spacing, origin, and size.
-   `.raw` stores the actual voxel data.
-   `annotations.csv` stores the center coordinates and diameters of true nodules.

The code uses SimpleITK to read `.mhd` files and automatically associate the corresponding `.raw` files.

## 3. Dataset Split in This Experiment

According to the final requirement of this experiment, the data split is:

| Usage | Subset | Number of CTs | Positive nodule samples | Negative samples | Total patches |
|----|---:|---:|---:|---:|---:|
| Training set | subset0, subset1, subset2 | 267 | 368 | 184 | 552 |
| Validation set | subset3 | 89 | 119 | 60 | 179 |
| Test set | subset4 | 89 | 128 | 64 | 192 |

Notes:

-   `subset0, subset1, subset2` are used for training.
-   `subset3` is used for validation and best checkpoint selection.
-   `subset4` is used for final testing.
-   `subset5` is not used in the final experiment.
-   Negative samples are randomly sampled by the code and are required to be at least `40 mm` away from known nodules, which avoids placing negative points too close to true nodules.

The corresponding code configuration is:

``` python
TRAIN_SUBSETS = (0, 1, 2)
VAL_SUBSETS = (3,)
TEST_SUBSETS = (4,)
```

## 4. Sample Construction

Each sample is represented as a `PatchSample`, which contains:

| Field | Meaning |
|----|----|
| `seriesuid` | CT series ID |
| `mhd_path` | Path to the corresponding CT `.mhd` file |
| `label` | Whether the sample is a nodule, where 1 means nodule and 0 means non-nodule |
| `subset` | Subset ID |
| `candidate_center_xyz` | Candidate center in world coordinates, in mm |
| `nodule_center_xyz` | True nodule center; `None` for negative samples |
| `diameter_mm` | True nodule diameter; 0 for negative samples |

Positive samples come from the true nodule annotations in `annotations.csv`.

Negative samples are generated as follows:

1.  Randomly select a voxel position in a CT volume from the specified subset.
2.  Convert the voxel position to world coordinates.
3.  Compute the distance from this point to all true nodule centers in the current CT.
4.  If the point is farther than `NEG_MIN_DIST_MM = 40.0` from every true nodule, it is used as a negative sample.

This reduces the risk of incorrectly treating regions near true nodules as negative samples.

## 5. CT Patch Cropping and Normalization

The final ultra-fast version uses the following patch size:

``` text
24 x 24 x 24
```

The command-line parameter is:

``` bash
--patch-size 24 24 24
```

Each patch is a three-dimensional local region cropped from the original CT volume, centered on the candidate point.

The intensity processing steps are:

1.  Clip CT HU values to:

``` text
HU_MIN = -1000
HU_MAX = 400
```

2.  Linearly normalize the values to `[0, 1]`.

3.  Map the values to `[-1, 1]`.

The code logic is:

``` python
patch = np.clip(patch, HU_MIN, HU_MAX)
patch = (patch - HU_MIN) / (HU_MAX - HU_MIN)
patch = patch * 2.0 - 1.0
```

The purpose is to:

-   Remove extreme HU values outside the lung window.
-   Keep the neural network input range more stable.
-   Accelerate training convergence.

## 6. Localization Label Definition

The localization task does not directly regress world coordinates. Instead, it regresses the normalized offset of the nodule center relative to the candidate patch center.

The model outputs:

``` text
pred_offset_x, pred_offset_y, pred_offset_z
```

Each offset is constrained to `[-1, 1]`.

The target offset is computed as:

``` text
target_offset_xyz = (target_idx_xyz - center_idx_xyz) / (patch_size_xyz / 2)
```

where:

-   `target_idx_xyz` is the voxel coordinate of the true nodule center.
-   `center_idx_xyz` is the voxel coordinate of the candidate patch center.
-   `patch_size_xyz / 2` is used for normalization.

During evaluation, the predicted offset is converted back to world coordinates:

``` text
pred_center_xyz = candidate_center_xyz
                  + pred_offset_xyz * (patch_size_xyz / 2) * spacing_xyz
```

Therefore, the final CSV files save:

-   Candidate point coordinates.
-   True nodule center coordinates.
-   Predicted nodule center coordinates.
-   Target offsets.
-   Predicted offsets.
-   Localization error in mm.

## 7. Model Architecture

The final model is a lightweight 3D CNN + small Transformer Encoder.

The model class name is:

``` python
Light3DCNNTransformer
```

### 7.1 Input

The input tensor shape is:

``` text
[batch_size, channels, depth, height, width]
```

In this experiment:

``` text
[B, 1, 24, 24, 24]
```

### 7.2 Lightweight 3D CNN Backbone

The CNN structure is:

``` text
ConvBlock3D(1 -> base_channels, stride=1)
ConvBlock3D(base_channels -> base_channels*2, stride=2)
DepthwiseSeparableConv3D(base_channels*2 -> base_channels*4, stride=2)
DepthwiseSeparableConv3D(base_channels*4 -> embed_dim, stride=2)
```

The parameters used in this experiment are:

``` text
base_channels = 4
embed_dim = 32
```

Therefore, the approximate channel progression is:

``` text
1 -> 4 -> 8 -> 16 -> 32
```

The spatial size changes as:

``` text
24 x 24 x 24
  -> 12 x 12 x 12
  -> 6 x 6 x 6
  -> 3 x 3 x 3
```

The final feature map produces `3 x 3 x 3 = 27` tokens, and each token has dimension `32`.

### 7.3 Depthwise Separable 3D Convolution

To accelerate training, this model replaces some standard 3D convolutions with `DepthwiseSeparableConv3D`.

A standard 3D convolution is computationally expensive, while a depthwise separable 3D convolution is divided into two steps:

1.  Depthwise 3D convolution: each channel is convolved independently.
2.  Pointwise 1x1x1 convolution: channel information is fused.

This greatly reduces the number of parameters and the amount of computation.

### 7.4 Transformer Encoder

The 3D feature map output by the CNN is flattened into a token sequence:

``` text
[B, C, D, H, W] -> [B, D*H*W, C]
```

In this experiment:

``` text
[B, 32, 3, 3, 3] -> [B, 27, 32]
```

Transformer configuration:

| Parameter            | Value |
|----------------------|------:|
| `embed_dim`          |    32 |
| `transformer_layers` |     1 |
| `transformer_heads`  |     2 |
| `dim_feedforward`    |    64 |
| `dropout`            |   0.1 |

The Transformer allows different spatial locations inside the patch to build global relationships, helping the model understand the context between the nodule and surrounding structures.

### 7.5 Output Heads

The model has two output heads.

1.  Classification head `cls_head`

``` text
Linear(embed_dim -> embed_dim/2)
GELU
Dropout
Linear(embed_dim/2 -> 1)
```

It outputs one logit. After sigmoid, the result is:

``` text
prob_nodule
```

2.  Localization head `loc_head`

``` text
Linear(embed_dim -> embed_dim/2)
GELU
Dropout
Linear(embed_dim/2 -> 3)
Tanh
```

It outputs:

``` text
pred_offset_x, pred_offset_y, pred_offset_z
```

Because `Tanh` is used at the end, the localization offsets are constrained to `[-1, 1]`.

## 8. Loss Function

This project uses a multi-task loss:

``` text
total_loss = cls_loss + loc_weight * loc_loss
```

### 8.1 Classification Loss

The classification loss is:

``` python
nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

Here, `pos_weight` is automatically computed according to the ratio between positive and negative samples in the training set.

In this experiment, the training set contains:

``` text
positive = 368
negative = 184
```

Because there are more positive samples than negative samples, the minimum `pos_weight` is set to `1.0`.

### 8.2 Localization Loss

The localization loss is:

``` python
SmoothL1Loss
```

Localization loss is computed only for positive samples. Negative samples do not have true nodule centers, so they do not participate in localization loss.

This is controlled by `loc_mask`:

``` text
loc_mask = 1 means positive sample and participates in localization loss
loc_mask = 0 means negative sample and does not participate in localization loss
```

The final parameter used in this experiment is:

``` text
loc_weight = 1.0
```

## 9. Training Acceleration Strategy

In the initial version, one epoch could take several minutes. To complete 80 epochs within a tight time limit, several acceleration strategies were applied.

### 9.1 Use a Smaller Patch

The patch size was reduced to:

``` text
24 x 24 x 24
```

Benefits:

-   Fewer input voxels.
-   Significantly lower 3D convolution computation.
-   Transformer token count is reduced to 27.

Trade-offs:

-   Smaller contextual range.
-   Potentially weaker representation for large nodules or boundary information.

### 9.2 Use a Lighter Model

The final model parameters are:

``` text
base_channels = 4
embed_dim = 32
transformer_layers = 1
transformer_heads = 2
batch_size = 64
```

Compared with the previous larger model, the computation is greatly reduced.

### 9.3 Skip 1 mm Resampling

The final version uses:

``` bash
--no-resample
```

This means patches are cropped directly using the original CT spacing, instead of resampling the whole CT volume to 1 mm isotropic spacing.

Benefits:

-   Avoids large-scale resampling after reading CT volumes.
-   Makes pre-caching faster.

Trade-offs:

-   Voxel spacing differs across CT scans.
-   Physical scale consistency is weaker than in the resampled version.

### 9.4 Pre-cache Patches to Disk

The final version enables:

``` bash
--precache-patches
--cache-float16
```

Before training, all 3D patches are cropped and saved to:

``` text
luna16_3d_outputs/patch_cache_ultrafast/
```

In this way, each epoch no longer repeatedly reads full CT volumes. It only reads already cropped small patches.

### 9.5 Load Cached Patches into Memory

The final version enables:

``` bash
--cache-in-memory
```

This loads `.npz` patches into RAM before training. During training, patches are read directly from memory, further reducing disk I/O.

### 9.6 Reduce the Negative Sample Ratio

This experiment uses:

``` bash
--negative-ratio 0.5
```

This means the number of negative samples is approximately half the number of positive samples. It reduces the total number of training samples and improves speed.

## 10. Running Environment

The Python interpreter used is:

``` text
/Applications/anaconda3/bin/python3
```

Main dependencies:

``` text
torch
SimpleITK
numpy
pandas
matplotlib
scikit-learn
tqdm
```

Dependency file:

``` text
requirements_3d.txt
```

The device is selected automatically:

``` python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

The actual device used in this experiment was:

``` text
mps
```

This is the Apple Silicon Metal backend.

## 11. Running Commands

### 11.1 80-epoch Training

Run:

``` bash
./run_80epoch_fast_training.sh
```

Internally, this script calls:

``` bash
./run_fast_training.sh
```

The main parameters currently used by `run_fast_training.sh` are:

``` bash
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

### 11.2 Evaluation Only, Without Retraining

If a checkpoint already exists, run the test-set evaluation and regenerate prediction images with:

``` bash
/Applications/anaconda3/bin/python3 train_luna16_3d_cnn_transformer.py --eval-only
```

This command reads:

``` text
luna16_3d_outputs/best_light3d_cnn_transformer_luna16.pth
```

and regenerates:

``` text
test_predictions_3d.csv
test_nodule_predictions_3d.csv
test_metrics_3d.json
test_nodule_visualizations/
```

## 12. Output Files

All main outputs are stored in:

``` text
luna16_3d_outputs/
```

### 12.1 Model File

``` text
luna16_3d_outputs/best_light3d_cnn_transformer_luna16.pth
```

This is the model checkpoint saved at the best validation AUC.

The best epoch in this experiment was:

``` text
epoch = 29
```

Best validation results:

| Metric                       |     Value |
|------------------------------|----------:|
| Best val AUC                 |    0.9548 |
| Best val accuracy            |    0.8771 |
| Best val F1                  |    0.9076 |
| Best val mean location error | 0.7429 mm |

### 12.2 Training History

``` text
luna16_3d_outputs/training_history_3d.csv
```

This file contains 80 rows, one row for each epoch.

Main columns:

| Column                    | Meaning                                         |
|---------------------------|-------------------------------------------------|
| `epoch`                   | Epoch number                                    |
| `lr`                      | Current learning rate                           |
| `train_loss`              | Total training loss                             |
| `train_accuracy`          | Training accuracy                               |
| `train_auc`               | Training AUC                                    |
| `train_f1`                | Training F1                                     |
| `val_loss`                | Total validation loss                           |
| `val_accuracy`            | Validation accuracy                             |
| `val_auc`                 | Validation AUC                                  |
| `val_f1`                  | Validation F1                                   |
| `val_loc_error_mean_mm`   | Mean localization error on the validation set   |
| `val_loc_error_median_mm` | Median localization error on the validation set |

### 12.3 Test Metrics

``` text
luna16_3d_outputs/test_metrics_3d.json
```

Final test-set results:

| Metric                  |     Value |
|-------------------------|----------:|
| Loss                    |    0.2760 |
| Classification loss     |    0.2754 |
| Location loss           |    0.0006 |
| Accuracy                |    0.9115 |
| Precision               |    0.9111 |
| Recall                  |    0.9609 |
| F1                      |    0.9354 |
| AUC                     |    0.9405 |
| Mean location error     | 0.6512 mm |
| Median location error   | 0.4761 mm |
| Saved prediction images |       128 |

### 12.4 Full Test Prediction Results

``` text
luna16_3d_outputs/test_predictions_3d.csv
```

This file contains all 192 samples in subset4, including both positive and negative samples.

Main columns:

| Column | Meaning |
|----|----|
| `seriesuid` | CT series ID |
| `label` | True label, where 1 means nodule and 0 means non-nodule |
| `prob_nodule` | Predicted probability of being a nodule |
| `pred_label` | Predicted label using threshold 0.5 |
| `diameter_mm` | True nodule diameter; 0 for negative samples |
| `candidate_x/y/z` | Candidate point center coordinates |
| `target_x/y/z` | True nodule center coordinates |
| `pred_x/y/z` | Predicted nodule center coordinates |
| `target_offset_x/y/z` | Target normalized offsets |
| `pred_offset_x/y/z` | Predicted normalized offsets |
| `loc_error_mm` | Localization error in mm |
| `visualization_path` | Path to the corresponding three-view nodule visualization image |

Negative samples do not have true nodule centers, so `loc_error_mm` is NaN.

### 12.5 Nodule-only Prediction Results

``` text
luna16_3d_outputs/test_nodule_predictions_3d.csv
```

This file is filtered from `test_predictions_3d.csv` and contains only true nodule samples, with 128 rows in total.

It is more suitable for checking:

-   The model probability for each true nodule.
-   Whether the nodule was missed.
-   The predicted nodule center coordinates.
-   The true nodule center coordinates.
-   The localization error.
-   The corresponding three-view image path.

Statistics of true nodule samples:

| Item                            | Value |
|---------------------------------|------:|
| True nodules in subset4         |   128 |
| Correctly classified as nodules |   123 |
| Missed as non-nodules           |     5 |

### 12.6 Three-view Nodule Visualization Images

``` text
luna16_3d_outputs/test_nodule_visualizations/
```

This experiment saved:

``` text
128 PNG images
```

Each image contains three views:

| View     | Meaning        |
|----------|----------------|
| Axial    | Axial plane    |
| Coronal  | Coronal plane  |
| Sagittal | Sagittal plane |

Markers in the images:

| Marker    | Meaning                 |
|-----------|-------------------------|
| Green `x` | True nodule center      |
| Red `+`   | Predicted nodule center |

The image title contains:

``` text
seriesuid
prob_nodule
pred label
diameter_mm
loc_error_mm
```

Example image:

![Accurate localization example](luna16_3d_outputs/test_nodule_visualizations/nodule_0006_1_3_6_1_4_1_14519_5_2_1_6279_6001_107351566259572521472765997306.png)

Prediction result of this sample:

``` text
prob = 0.971
pred = Nodule
diameter = 4.21 mm
loc_error = 0.03 mm
```

Missed nodule example:

![Missed nodule example](luna16_3d_outputs/test_nodule_visualizations/nodule_0037_1_3_6_1_4_1_14519_5_2_1_6279_6001_161855583909753609742728521805.png)

Prediction result of this sample:

``` text
prob = 0.095
pred = Non-nodule
diameter = 7.43 mm
loc_error = 1.91 mm
```

## 13. Training Curves and Evaluation Figures

### 13.1 Loss Curve

``` text
luna16_3d_outputs/loss_curve_3d.png
```

![Loss curve](luna16_3d_outputs/loss_curve_3d.png)

This figure shows the total loss of the training set and validation set across epochs.

### 13.2 Accuracy Curve

``` text
luna16_3d_outputs/accuracy_curve_3d.png
```

![Accuracy curve](luna16_3d_outputs/accuracy_curve_3d.png)

This figure shows the training accuracy and validation accuracy across epochs.

### 13.3 AUC Curve

``` text
luna16_3d_outputs/auc_curve_3d.png
```

![AUC curve](luna16_3d_outputs/auc_curve_3d.png)

This figure shows the training AUC and validation AUC across epochs.

### 13.4 Localization Error Curve

``` text
luna16_3d_outputs/location_error_curve_3d.png
```

![Location error curve](luna16_3d_outputs/location_error_curve_3d.png)

This figure shows the mean localization error on the validation set across epochs.

### 13.5 Confusion Matrix

``` text
luna16_3d_outputs/confusion_matrix_3d.png
```

![Confusion matrix](luna16_3d_outputs/confusion_matrix_3d.png)

Test-set confusion matrix:

| True class / Predicted class | Pred Negative | Pred Positive |
|------------------------------|--------------:|--------------:|
| Actual Negative              |            52 |            12 |
| Actual Positive              |             5 |           123 |

Interpretation:

-   52 negative samples were correctly classified as negative.
-   12 negative samples were misclassified as nodules.
-   5 true nodules were missed.
-   123 true nodules were correctly classified as nodules.

### 13.6 ROC Curve

``` text
luna16_3d_outputs/roc_curve_3d.png
```

![ROC curve](luna16_3d_outputs/roc_curve_3d.png)

Test-set AUC:

``` text
0.9405
```

### 13.7 PR Curve

``` text
luna16_3d_outputs/pr_curve_3d.png
```

![PR curve](luna16_3d_outputs/pr_curve_3d.png)

The PR curve shows the relationship between precision and recall. For pulmonary nodule detection, recall is important because missing a nodule is usually more serious than reporting some additional candidates.

## 14. Interpretation of the Results

### 14.1 Classification Performance

Test-set results:

``` text
Accuracy = 0.9115
Precision = 0.9111
Recall = 0.9609
F1 = 0.9354
AUC = 0.9405
```

These metrics indicate that the model has good nodule recognition ability on subset4.

The recall is `0.9609`, meaning that most true nodules were detected:

``` text
123 / 128
```

Number of missed nodules:

``` text
5
```

The precision is `0.9111`, meaning that most samples predicted as nodules are true nodules, although some false positives remain.

### 14.2 Localization Performance

Localization error:

``` text
Mean location error = 0.6512 mm
Median location error = 0.4761 mm
```

This indicates that, for patches labeled as true nodules, the predicted center is usually very close to the true center.

One of the samples with the smallest error:

``` text
loc_error = 0.0271 mm
```

One of the samples with larger error:

``` text
loc_error = 3.6317 mm
```

It should be noted that the current localization is local regression inside cropped candidate patches. It is not global nodule search over the whole CT volume.

## 15. Code Structure

The core code is located in:

``` text
train_luna16_3d_cnn_transformer.py
```

The main modules are described below.

### 15.1 Data Structures

``` python
VolumeInfo
PatchSample
VolumeData
```

These classes store CT metadata, patch sample information, and volume data.

### 15.2 Coordinate Conversion

``` python
world_to_index_xyz
index_to_world_xyz
```

These functions convert between world coordinates and voxel coordinates.

LUNA16 annotations use world coordinates, while CT array cropping uses voxel coordinates. Therefore, these functions are critical.

### 15.3 CT Loading

``` python
load_volume
VolumeCache
```

These functions are responsible for reading `.mhd` CT files and caching volume data when necessary, avoiding repeated reads.

### 15.4 Patch Construction

``` python
build_patch_item
crop_patch_zyx
normalize_patch_hu
augment_patch
```

These functions are responsible for:

-   Cropping patches according to candidate centers.
-   Normalizing HU values.
-   Applying lightweight augmentation such as flipping and brightness perturbation to training samples.
-   Generating classification labels and localization labels.

### 15.5 Dataset

``` python
Luna3DPatchDataset
CachedLuna3DPatchDataset
```

`CachedLuna3DPatchDataset` is one of the key acceleration components in this experiment. It reads pre-cropped patches and can load them into memory.

### 15.6 Model

``` python
ConvBlock3D
DepthwiseSeparableConv3D
Light3DCNNTransformer
```

The model consists of a lightweight 3D CNN, a small Transformer, a classification head, and a localization head.

### 15.7 Training and Evaluation

``` python
train_one_epoch
evaluate
multitask_loss
summarize_epoch
```

`evaluate` performs:

-   Classification metric calculation.
-   Localization error calculation.
-   Prediction CSV export.
-   Three-view nodule visualization image export.

### 15.8 Visualization Output

``` python
save_plots
save_nodule_prediction_image
```

`save_plots` outputs the training curves, confusion matrix, ROC curve, and PR curve.

`save_nodule_prediction_image` outputs a three-view image for each true nodule and marks both the true position and the predicted position.

## 16. Key Parameter Summary

| Parameter | Value | Description |
|----|---:|----|
| `patch_size` | 24 24 24 | Input 3D patch size |
| `negative_ratio` | 0.5 | Number of negative samples is approximately half the number of positive samples |
| `positive_jitter_mm` | 2.0 | Random perturbation of positive centers during training |
| `batch_size` | 64 | Batch size |
| `epochs` | 80 | Number of training epochs |
| `lr` | 5e-4 | Learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `base_channels` | 4 | Base number of CNN channels |
| `embed_dim` | 32 | Transformer token dimension |
| `transformer_layers` | 1 | Number of Transformer layers |
| `transformer_heads` | 2 | Number of attention heads |
| `dropout` | 0.1 | Dropout |
| `loc_weight` | 1.0 | Localization loss weight |
| `cache_float16` | True | Use float16 for cached patches |
| `cache_in_memory` | True | Load patches into memory before training |
| `no_resample` | True | Do not perform 1 mm resampling |

## 17. Explanation of Output CSV Examples

One well-predicted sample in `test_nodule_predictions_3d.csv`:

``` text
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

Interpretation:

-   The model predicts this patch as a nodule with probability `0.970727`.
-   `pred_label = 1`, meaning it is classified as a nodule.
-   The distance between the predicted center and the true center is about `0.027 mm`.
-   This is a very accurate localization sample.

A missed nodule sample:

``` text
seriesuid = 1.3.6.1.4.1.14519.5.2.1.6279.6001.161855583909753609742728521805
prob_nodule = 0.094697
pred_label = 0
diameter_mm = 7.428431
loc_error_mm = 1.908435
```

Interpretation:

-   This sample is truly a nodule.
-   However, the model gives it a nodule probability of only `0.094697`.
-   Since this is lower than the threshold `0.5`, it is classified as a non-nodule.
-   This is a missed detection.

## 18. Advantages of the Current Method

1.  Fast training.

Through small patches, a lightweight CNN, a small Transformer, pre-caching, and memory caching, 80 epochs can be completed in a short time.

2.  Complete outputs.

This project outputs:

-   Loss curve.
-   Accuracy curve.
-   AUC curve.
-   Localization error curve.
-   Confusion matrix.
-   ROC curve.
-   PR curve.
-   Full test prediction CSV.
-   Nodule-only prediction CSV.
-   Three-view nodule visualization images.

3.  Classification and localization are both included.

The model does not only determine whether a patch contains a nodule. It also outputs the nodule center offset and the final world coordinates.

4.  The code can directly reproduce the experiment.

Training, evaluation, and visualization are all integrated into one script, and the VS Code launch configuration is already prepared.

## 19. Limitations of the Current Method

1.  It is not a complete CT-level detection system.

The current model is based on candidate patches. It does not search the whole CT volume for all candidate nodules.

2.  The patch is small.

The `24 x 24 x 24` patch trains very quickly, but it contains less context. It may be less stable for blurred boundaries, large nodules, or samples with complex surrounding structures.

3.  1 mm isotropic resampling is not performed.

This improves speed, but voxel scale may differ across CT scans.

4.  The number of negative samples is small.

The current `negative_ratio = 0.5` improves speed, but in real clinical screening, negative candidates are much more numerous than positive candidates.

5.  The test set only uses subset4.

The current results only represent performance on subset4 under this split. Results may change if more subsets or cross-validation are used.

6.  Visualization images are local patch-level views.

The three-view images show the model input patches, not complete CT slices.

## 20. File Index

| File or directory | Description |
|----|----|
| `train_luna16_3d_cnn_transformer.py` | Main training, evaluation, and visualization script |
| `luna16_3d_outputs/best_light3d_cnn_transformer_luna16.pth` | Best model checkpoint |
| `luna16_3d_outputs/training_history_3d.csv` | Training history |
| `luna16_3d_outputs/test_metrics_3d.json` | Test metrics |
| `luna16_3d_outputs/test_predictions_3d.csv` | Full test predictions |
| `luna16_3d_outputs/test_nodule_predictions_3d.csv` | True nodule prediction summary |
| `luna16_3d_outputs/test_nodule_visualizations/` | Three-view images of true nodules |
| `luna16_3d_outputs/loss_curve_3d.png` | Loss curve |
| `luna16_3d_outputs/accuracy_curve_3d.png` | Accuracy curve |
| `luna16_3d_outputs/auc_curve_3d.png` | AUC curve |
| `luna16_3d_outputs/location_error_curve_3d.png` | Localization error curve |
| `luna16_3d_outputs/confusion_matrix_3d.png` | Confusion matrix |
| `luna16_3d_outputs/roc_curve_3d.png` | ROC curve |
| `luna16_3d_outputs/pr_curve_3d.png` | PR curve |

## 21. Conclusion

This experiment completed a lightweight 3D CNN + small Transformer Encoder multi-task model based on LUNA16. The model can determine whether a candidate 3D CT patch contains a pulmonary nodule and output the nodule center location.

Under the final split:

``` text
train = subset0, subset1, subset2
val = subset3
test = subset4
```

the model achieved:

``` text
Accuracy = 0.9115
F1 = 0.9354
AUC = 0.9405
Mean location error = 0.6512 mm
Median location error = 0.4761 mm
```

At the same time, the project has generated complete training curves, evaluation figures, prediction CSV files, and 128 three-view nodule visualization images. Overall, this project forms a complete small-scale research workflow that can train, evaluate, visualize, and reproduce the experimental results.

It should be emphasized that this model is still a candidate-patch-level research model and should not be directly used for clinical diagnosis. To develop it into a complete pulmonary nodule detection system, candidate generation, full-CT scanning, multi-scale detection, stricter cross-validation, and clinical-level error analysis are still required.

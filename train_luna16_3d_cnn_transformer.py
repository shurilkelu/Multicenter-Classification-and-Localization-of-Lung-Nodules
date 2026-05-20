"""
Train a lightweight 3D CNN + small Transformer Encoder model on LUNA16.

Split policy:
- subset0, subset1, subset2 -> train
- subset3 -> validation/evaluation
- subset4 -> test

Task:
- classify whether a 3D candidate patch contains a nodule
- regress the nodule center offset inside the patch for positive samples

The location head predicts normalized (dx, dy, dz) offsets relative to the
candidate patch center. During evaluation those offsets are converted back to
LUNA16 world coordinates in millimeters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/Users/wangshang/LUNA16")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "luna16_3d_outputs"

TRAIN_SUBSETS = (0, 1, 2)
VAL_SUBSETS = (3,)
TEST_SUBSETS = (4,)

SEED = 42
HU_MIN = -1000.0
HU_MAX = 400.0
NEG_MIN_DIST_MM = 40.0


@dataclass(frozen=True)
class VolumeInfo:
    path: str
    size_xyz: tuple[int, int, int]
    origin_xyz: tuple[float, float, float]
    spacing_xyz: tuple[float, float, float]
    direction: tuple[float, ...]


@dataclass(frozen=True)
class PatchSample:
    seriesuid: str
    mhd_path: str
    label: int
    subset: int
    candidate_center_xyz: tuple[float, float, float]
    nodule_center_xyz: tuple[float, float, float] | None
    diameter_mm: float


@dataclass
class VolumeData:
    array_zyx: np.ndarray
    origin_xyz: np.ndarray
    spacing_xyz: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_int_tuple(values: Iterable[int], length: int, name: str) -> tuple[int, ...]:
    parsed = tuple(int(v) for v in values)
    if len(parsed) != length:
        raise ValueError(f"{name} must contain {length} values.")
    return parsed


def parse_float_tuple(values: Iterable[float], length: int, name: str) -> tuple[float, ...]:
    parsed = tuple(float(v) for v in values)
    if len(parsed) != length:
        raise ValueError(f"{name} must contain {length} values.")
    return parsed


def subset_name(subset_id: int) -> str:
    return f"subset{subset_id}"


def find_mhd_files(data_dir: Path, subsets: Iterable[int]) -> dict[str, str]:
    mhd_dict: dict[str, str] = {}
    for subset_id in subsets:
        root = data_dir / subset_name(subset_id)
        if not root.exists():
            continue
        for mhd_path in sorted(root.rglob("*.mhd")):
            mhd_dict[mhd_path.stem] = str(mhd_path)
    return mhd_dict


def get_subset_id(path: str | Path) -> int:
    for part in Path(path).parts:
        if part.startswith("subset") and part[6:].isdigit():
            return int(part[6:])
    raise ValueError(f"Cannot infer subset id from path: {path}")


def read_image_info(path: str | Path) -> VolumeInfo:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return VolumeInfo(
        path=str(path),
        size_xyz=tuple(int(v) for v in reader.GetSize()),
        origin_xyz=tuple(float(v) for v in reader.GetOrigin()),
        spacing_xyz=tuple(float(v) for v in reader.GetSpacing()),
        direction=tuple(float(v) for v in reader.GetDirection()),
    )


def effective_resampled_info(info: VolumeInfo, spacing_xyz: tuple[float, float, float] | None) -> VolumeInfo:
    if spacing_xyz is None:
        return info

    new_size = []
    for size, old_spacing, new_spacing in zip(info.size_xyz, info.spacing_xyz, spacing_xyz):
        new_size.append(max(1, int(round(size * old_spacing / new_spacing))))

    return VolumeInfo(
        path=info.path,
        size_xyz=tuple(new_size),
        origin_xyz=info.origin_xyz,
        spacing_xyz=spacing_xyz,
        direction=info.direction,
    )


def world_to_index_xyz(
    world_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    if direction is None:
        direction = np.eye(3, dtype=np.float64)
    relative = world_xyz.astype(np.float64) - origin_xyz.astype(np.float64)
    return np.linalg.inv(direction).dot(relative) / spacing_xyz.astype(np.float64)


def index_to_world_xyz(
    index_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    if direction is None:
        direction = np.eye(3, dtype=np.float64)
    physical = direction.dot(index_xyz.astype(np.float64) * spacing_xyz.astype(np.float64))
    return physical + origin_xyz.astype(np.float64)


def load_volume(
    mhd_path: str | Path,
    resample_spacing_xyz: tuple[float, float, float] | None,
) -> VolumeData:
    image = sitk.ReadImage(str(mhd_path))
    if resample_spacing_xyz is not None:
        old_size = np.array(image.GetSize(), dtype=np.int64)
        old_spacing = np.array(image.GetSpacing(), dtype=np.float64)
        new_spacing = np.array(resample_spacing_xyz, dtype=np.float64)
        new_size = np.maximum(1, np.round(old_size * old_spacing / new_spacing).astype(np.int64))

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(tuple(float(v) for v in new_spacing))
        resampler.SetSize([int(v) for v in new_size])
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(HU_MIN)
        image = resampler.Execute(image)

    array_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    return VolumeData(
        array_zyx=array_zyx,
        origin_xyz=np.array(image.GetOrigin(), dtype=np.float64),
        spacing_xyz=np.array(image.GetSpacing(), dtype=np.float64),
    )


class VolumeCache:
    def __init__(self, max_items: int, resample_spacing_xyz: tuple[float, float, float] | None):
        self.max_items = max_items
        self.resample_spacing_xyz = resample_spacing_xyz
        self.cache: OrderedDict[str, VolumeData] = OrderedDict()

    def get(self, seriesuid: str, mhd_path: str) -> VolumeData:
        if seriesuid in self.cache:
            self.cache.move_to_end(seriesuid)
            return self.cache[seriesuid]

        volume = load_volume(mhd_path, self.resample_spacing_xyz)
        self.cache[seriesuid] = volume
        self.cache.move_to_end(seriesuid)

        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

        return volume


def build_patch_item(
    sample: PatchSample,
    volume_cache: VolumeCache,
    patch_size_zyx: tuple[int, int, int],
    train: bool,
    positive_jitter_mm: float,
    seed: int,
) -> dict[str, np.ndarray | str | float]:
    rng = random.Random(seed)
    patch_size_xyz = np.array(
        [patch_size_zyx[2], patch_size_zyx[1], patch_size_zyx[0]],
        dtype=np.float32,
    )

    volume = volume_cache.get(sample.seriesuid, sample.mhd_path)
    direction = np.eye(3, dtype=np.float64)

    candidate_center_xyz = np.array(sample.candidate_center_xyz, dtype=np.float64)
    if train and sample.label == 1 and positive_jitter_mm > 0:
        jitter = np.array(
            [rng.uniform(-positive_jitter_mm, positive_jitter_mm) for _ in range(3)],
            dtype=np.float64,
        )
        candidate_center_xyz = candidate_center_xyz + jitter

    center_idx_xyz = world_to_index_xyz(
        candidate_center_xyz,
        volume.origin_xyz,
        volume.spacing_xyz,
        direction,
    )
    center_idx_zyx = np.array([center_idx_xyz[2], center_idx_xyz[1], center_idx_xyz[0]], dtype=np.float64)

    patch = crop_patch_zyx(volume.array_zyx, center_idx_zyx, patch_size_zyx)
    patch = normalize_patch_hu(patch)

    target_offset_xyz = np.zeros(3, dtype=np.float32)
    loc_mask = np.float32(0.0)
    target_center_xyz = np.array(candidate_center_xyz, dtype=np.float64)
    if sample.label == 1 and sample.nodule_center_xyz is not None:
        loc_mask = np.float32(1.0)
        target_center_xyz = np.array(sample.nodule_center_xyz, dtype=np.float64)
        target_idx_xyz = world_to_index_xyz(
            target_center_xyz,
            volume.origin_xyz,
            volume.spacing_xyz,
            direction,
        )
        target_offset_xyz = ((target_idx_xyz - center_idx_xyz) / (patch_size_xyz / 2.0)).astype(np.float32)
        target_offset_xyz = np.clip(target_offset_xyz, -1.0, 1.0)

    return {
        "image": patch[None, ...].astype(np.float32),
        "label": np.array(float(sample.label), dtype=np.float32),
        "target_offset": target_offset_xyz.astype(np.float32),
        "loc_mask": np.array(loc_mask, dtype=np.float32),
        "seriesuid": sample.seriesuid,
        "candidate_center": candidate_center_xyz.astype(np.float32),
        "target_center": target_center_xyz.astype(np.float32),
        "spacing": volume.spacing_xyz.astype(np.float32),
        "diameter_mm": np.array(float(sample.diameter_mm), dtype=np.float32),
    }


def normalize_patch_hu(patch: np.ndarray) -> np.ndarray:
    patch = np.clip(patch, HU_MIN, HU_MAX)
    patch = (patch - HU_MIN) / (HU_MAX - HU_MIN)
    return (patch * 2.0 - 1.0).astype(np.float32)


def crop_patch_zyx(
    volume_zyx: np.ndarray,
    center_zyx: np.ndarray,
    patch_size_zyx: tuple[int, int, int],
    pad_value: float = HU_MIN,
) -> np.ndarray:
    center = np.round(center_zyx).astype(np.int64)
    size = np.array(patch_size_zyx, dtype=np.int64)
    start = center - size // 2
    end = start + size
    patch = np.full(tuple(size), pad_value, dtype=np.float32)

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.array(volume_zyx.shape, dtype=np.int64))
    if np.any(src_end <= src_start):
        return patch

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    patch[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = volume_zyx[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    return patch


def augment_patch(
    patch_zyx: np.ndarray,
    target_offset_xyz: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    offset = target_offset_xyz.astype(np.float32).copy()

    if rng.random() < 0.5:
        patch_zyx = patch_zyx[:, :, ::-1]
        offset[0] *= -1.0
    if rng.random() < 0.5:
        patch_zyx = patch_zyx[:, ::-1, :]
        offset[1] *= -1.0
    if rng.random() < 0.25:
        patch_zyx = patch_zyx[::-1, :, :]
        offset[2] *= -1.0

    scale = rng.uniform(0.9, 1.1)
    shift = rng.uniform(-0.08, 0.08)
    patch_zyx = np.clip(patch_zyx * scale + shift, -1.0, 1.0)
    return np.ascontiguousarray(patch_zyx), offset


class Luna3DPatchDataset(Dataset):
    def __init__(
        self,
        samples: list[PatchSample],
        patch_size_zyx: tuple[int, int, int],
        resample_spacing_xyz: tuple[float, float, float] | None,
        cache_size: int,
        train: bool,
        positive_jitter_mm: float,
        seed: int,
    ):
        self.samples = samples
        self.patch_size_zyx = patch_size_zyx
        self.patch_size_xyz = np.array(
            [patch_size_zyx[2], patch_size_zyx[1], patch_size_zyx[0]],
            dtype=np.float32,
        )
        self.resample_spacing_xyz = resample_spacing_xyz
        self.cache = VolumeCache(cache_size, resample_spacing_xyz)
        self.train = train
        self.positive_jitter_mm = positive_jitter_mm
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | float]:
        sample = self.samples[idx]
        item_seed = self.seed + idx + random.randint(0, 1_000_000) if self.train else self.seed + idx
        item = build_patch_item(
            sample=sample,
            volume_cache=self.cache,
            patch_size_zyx=self.patch_size_zyx,
            train=self.train,
            positive_jitter_mm=self.positive_jitter_mm,
            seed=item_seed,
        )
        patch = item["image"][0]
        target_offset_xyz = item["target_offset"]

        if self.train:
            rng = random.Random(item_seed)
            patch, target_offset_xyz = augment_patch(patch, target_offset_xyz, rng)

        return {
            "image": torch.from_numpy(patch[None, ...].copy()),
            "label": torch.tensor(float(item["label"]), dtype=torch.float32),
            "target_offset": torch.from_numpy(target_offset_xyz),
            "loc_mask": torch.tensor(float(item["loc_mask"]), dtype=torch.float32),
            "seriesuid": str(item["seriesuid"]),
            "candidate_center": torch.from_numpy(item["candidate_center"]),
            "target_center": torch.from_numpy(item["target_center"]),
            "spacing": torch.from_numpy(item["spacing"]),
            "diameter_mm": torch.tensor(float(item["diameter_mm"]), dtype=torch.float32),
        }


class CachedLuna3DPatchDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, str | int | float]],
        patch_size_zyx: tuple[int, int, int],
        train: bool,
        seed: int,
        cache_in_memory: bool,
    ):
        self.items = items
        self.patch_size_zyx = patch_size_zyx
        self.train = train
        self.seed = seed
        self.memory_items: list[dict[str, np.ndarray]] | None = None

        if cache_in_memory:
            self.memory_items = []
            for meta in tqdm(items, desc="Loading cached patches into RAM", leave=False):
                with np.load(str(meta["path"])) as data:
                    self.memory_items.append(
                        {
                            "image": data["image"].copy(),
                            "label": np.array(data["label"]).copy(),
                            "target_offset": data["target_offset"].copy(),
                            "loc_mask": np.array(data["loc_mask"]).copy(),
                            "candidate_center": data["candidate_center"].copy(),
                            "target_center": data["target_center"].copy(),
                            "spacing": data["spacing"].copy(),
                            "diameter_mm": np.array(data["diameter_mm"]).copy(),
                        }
                    )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | float]:
        meta = self.items[idx]
        if self.memory_items is not None:
            return self._make_item(idx, meta, self.memory_items[idx])

        with np.load(str(meta["path"])) as data:
            arrays = {
                "image": data["image"],
                "label": data["label"],
                "target_offset": data["target_offset"],
                "loc_mask": data["loc_mask"],
                "candidate_center": data["candidate_center"],
                "target_center": data["target_center"],
                "spacing": data["spacing"],
                "diameter_mm": data["diameter_mm"],
            }
            return self._make_item(idx, meta, arrays)

    def _make_item(
        self,
        idx: int,
        meta: dict[str, str | int | float],
        arrays: dict[str, np.ndarray],
    ) -> dict[str, torch.Tensor | str | float]:
        image = arrays["image"].astype(np.float32, copy=False)
        patch = image[0]
        target_offset = arrays["target_offset"].astype(np.float32, copy=True)

        if self.train:
            rng = random.Random(self.seed + idx + random.randint(0, 1_000_000))
            patch, target_offset = augment_patch(patch, target_offset, rng)

        return {
            "image": torch.from_numpy(patch[None, ...].copy()),
            "label": torch.tensor(float(arrays["label"]), dtype=torch.float32),
            "target_offset": torch.from_numpy(target_offset),
            "loc_mask": torch.tensor(float(arrays["loc_mask"]), dtype=torch.float32),
            "seriesuid": str(meta["seriesuid"]),
            "candidate_center": torch.from_numpy(arrays["candidate_center"].astype(np.float32)),
            "target_center": torch.from_numpy(arrays["target_center"].astype(np.float32)),
            "spacing": torch.from_numpy(arrays["spacing"].astype(np.float32)),
            "diameter_mm": torch.tensor(float(arrays["diameter_mm"]), dtype=torch.float32),
        }


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def downsampled_size(size: int, downsamples: int = 3) -> int:
    for _ in range(downsamples):
        size = math.ceil(size / 2)
    return size


class Light3DCNNTransformer(nn.Module):
    def __init__(
        self,
        patch_size_zyx: tuple[int, int, int],
        base_channels: int = 16,
        embed_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.patch_size_zyx = patch_size_zyx

        self.cnn = nn.Sequential(
            ConvBlock3D(1, base_channels, stride=1),
            ConvBlock3D(base_channels, base_channels * 2, stride=2),
            DepthwiseSeparableConv3D(base_channels * 2, base_channels * 4, stride=2),
            DepthwiseSeparableConv3D(base_channels * 4, embed_dim, stride=2),
        )

        token_grid = tuple(downsampled_size(v) for v in patch_size_zyx)
        num_tokens = int(np.prod(token_grid))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self.loc_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.cnn(x)
        tokens = features.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.position_embedding.shape[1]:
            raise ValueError(
                f"Unexpected token count {tokens.shape[1]}; "
                f"expected {self.position_embedding.shape[1]} for patch size {self.patch_size_zyx}."
            )
        tokens = tokens + self.position_embedding
        encoded = self.transformer(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        cls_logit = self.cls_head(pooled).squeeze(1)
        loc_offset = self.loc_head(pooled)
        return cls_logit, loc_offset


def collect_samples(
    data_dir: Path,
    split_subsets: dict[str, tuple[int, ...]],
    negative_ratio: float,
    seed: int,
    resample_spacing_xyz: tuple[float, float, float] | None,
) -> dict[str, list[PatchSample]]:
    annotations_path = data_dir / "annotations.csv"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations.csv: {annotations_path}")

    all_subsets = tuple(sorted({sid for subset_ids in split_subsets.values() for sid in subset_ids}))
    missing_subsets = [subset_name(sid) for sid in all_subsets if not (data_dir / subset_name(sid)).exists()]
    if missing_subsets:
        raise FileNotFoundError(
            "Missing required LUNA16 subset directories:\n- " + "\n- ".join(missing_subsets)
        )

    mhd_dict = find_mhd_files(data_dir, all_subsets)
    if not mhd_dict:
        raise FileNotFoundError(f"No .mhd files found under {data_dir}")

    annotations = pd.read_csv(annotations_path)
    annotations = annotations[annotations["seriesuid"].isin(mhd_dict.keys())].reset_index(drop=True)

    annotations_by_series = {
        seriesuid: group.reset_index(drop=True)
        for seriesuid, group in annotations.groupby("seriesuid")
    }

    samples_by_split: dict[str, list[PatchSample]] = {}
    rng = random.Random(seed)

    for split_name, subset_ids in split_subsets.items():
        split_mhd = {
            seriesuid: path
            for seriesuid, path in mhd_dict.items()
            if get_subset_id(path) in subset_ids
        }
        split_annotations = annotations[annotations["seriesuid"].isin(split_mhd.keys())]

        positives: list[PatchSample] = []
        for _, row in split_annotations.iterrows():
            seriesuid = str(row["seriesuid"])
            center_xyz = (
                float(row["coordX"]),
                float(row["coordY"]),
                float(row["coordZ"]),
            )
            positives.append(
                PatchSample(
                    seriesuid=seriesuid,
                    mhd_path=split_mhd[seriesuid],
                    label=1,
                    subset=get_subset_id(split_mhd[seriesuid]),
                    candidate_center_xyz=center_xyz,
                    nodule_center_xyz=center_xyz,
                    diameter_mm=float(row["diameter_mm"]),
                )
            )

        target_negatives = int(round(len(positives) * negative_ratio))
        negatives = generate_negative_samples(
            split_mhd=split_mhd,
            annotations_by_series=annotations_by_series,
            target_count=target_negatives,
            seed=rng.randint(0, 2**31 - 1),
            resample_spacing_xyz=resample_spacing_xyz,
        )

        samples = positives + negatives
        rng.shuffle(samples)
        samples_by_split[split_name] = samples

        print(
            f"{split_name}: subsets={subset_ids}, CT={len(split_mhd)}, "
            f"positive={len(positives)}, negative={len(negatives)}, total={len(samples)}"
        )

    return samples_by_split


def generate_negative_samples(
    split_mhd: dict[str, str],
    annotations_by_series: dict[str, pd.DataFrame],
    target_count: int,
    seed: int,
    resample_spacing_xyz: tuple[float, float, float] | None,
) -> list[PatchSample]:
    if target_count <= 0:
        return []
    if not split_mhd:
        raise ValueError("Cannot generate negatives for an empty split.")

    rng = random.Random(seed)
    seriesuids = sorted(split_mhd.keys())
    volume_info_cache: dict[str, VolumeInfo] = {}
    negatives: list[PatchSample] = []
    attempts = 0
    max_attempts = max(1000, target_count * 80)

    pbar = tqdm(total=target_count, desc="Sampling negatives", leave=False)
    while len(negatives) < target_count and attempts < max_attempts:
        attempts += 1
        seriesuid = rng.choice(seriesuids)
        mhd_path = split_mhd[seriesuid]

        if seriesuid not in volume_info_cache:
            raw_info = read_image_info(mhd_path)
            volume_info_cache[seriesuid] = effective_resampled_info(raw_info, resample_spacing_xyz)
        info = volume_info_cache[seriesuid]

        idx_xyz = np.array(
            [
                rng.uniform(0, max(size - 1, 0))
                for size in info.size_xyz
            ],
            dtype=np.float64,
        )
        center_xyz = index_to_world_xyz(
            idx_xyz,
            np.array(info.origin_xyz, dtype=np.float64),
            np.array(info.spacing_xyz, dtype=np.float64),
            np.eye(3, dtype=np.float64),
        )

        if not is_far_from_known_nodules(seriesuid, center_xyz, annotations_by_series):
            continue

        negatives.append(
            PatchSample(
                seriesuid=seriesuid,
                mhd_path=mhd_path,
                label=0,
                subset=get_subset_id(mhd_path),
                candidate_center_xyz=tuple(float(v) for v in center_xyz),
                nodule_center_xyz=None,
                diameter_mm=0.0,
            )
        )
        pbar.update(1)

    pbar.close()

    if len(negatives) < target_count:
        print(f"Warning: requested {target_count} negatives but sampled {len(negatives)}.")
    return negatives


def is_far_from_known_nodules(
    seriesuid: str,
    center_xyz: np.ndarray,
    annotations_by_series: dict[str, pd.DataFrame],
    min_dist_mm: float = NEG_MIN_DIST_MM,
) -> bool:
    rows = annotations_by_series.get(seriesuid)
    if rows is None or rows.empty:
        return True

    nodule_xyz = rows[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)
    distances = np.linalg.norm(nodule_xyz - center_xyz[None, :], axis=1)
    return bool(np.all(distances >= min_dist_mm))


def patch_cache_config(
    split: str,
    patch_size_zyx: tuple[int, int, int],
    resample_spacing_xyz: tuple[float, float, float] | None,
    positive_jitter_mm: float,
    train_cache_copies: int,
    seed: int,
    cache_float16: bool,
) -> dict[str, object]:
    return {
        "split": split,
        "patch_size_zyx": list(patch_size_zyx),
        "resample_spacing_xyz": None if resample_spacing_xyz is None else list(resample_spacing_xyz),
        "positive_jitter_mm": positive_jitter_mm if split == "train" else 0.0,
        "train_cache_copies": train_cache_copies if split == "train" else 1,
        "seed": seed,
        "cache_float16": cache_float16,
    }


def build_or_load_patch_cache(
    samples_by_split: dict[str, list[PatchSample]],
    patch_size_zyx: tuple[int, int, int],
    resample_spacing_xyz: tuple[float, float, float] | None,
    cache_dir: Path,
    cache_size: int,
    positive_jitter_mm: float,
    train_cache_copies: int,
    seed: int,
    rebuild: bool,
    cache_float16: bool,
) -> dict[str, list[dict[str, str | int | float]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_by_split: dict[str, list[dict[str, str | int | float]]] = {}

    for split, samples in samples_by_split.items():
        split_dir = cache_dir / split
        manifest_path = split_dir / "manifest.json"
        config = patch_cache_config(
            split,
            patch_size_zyx,
            resample_spacing_xyz,
            positive_jitter_mm,
            train_cache_copies,
            seed,
            cache_float16,
        )

        if not rebuild and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("config") == config:
                items = manifest.get("items", [])
                if items and all(Path(str(item["path"])).exists() for item in items):
                    print(f"Using cached {split} patches: {len(items)}")
                    cached_by_split[split] = items
                    continue

        split_dir.mkdir(parents=True, exist_ok=True)
        for old_file in split_dir.glob("*.npz"):
            old_file.unlink()

        copies = max(1, train_cache_copies if split == "train" else 1)
        volume_cache = VolumeCache(cache_size, resample_spacing_xyz)
        items: list[dict[str, str | int | float]] = []

        indexed_samples = sorted(enumerate(samples), key=lambda item: (item[1].mhd_path, item[0]))
        total = len(indexed_samples) * copies
        pbar = tqdm(total=total, desc=f"Caching {split} patches")
        for sample_idx, sample in indexed_samples:
            for copy_idx in range(copies):
                item_seed = seed + sample_idx * 1009 + copy_idx * 9176
                item = build_patch_item(
                    sample=sample,
                    volume_cache=volume_cache,
                    patch_size_zyx=patch_size_zyx,
                    train=split == "train",
                    positive_jitter_mm=positive_jitter_mm if split == "train" else 0.0,
                    seed=item_seed,
                )

                image = item["image"]
                if cache_float16:
                    image = image.astype(np.float16)

                cache_idx = len(items)
                file_path = split_dir / f"{split}_{cache_idx:05d}.npz"
                np.savez(
                    file_path,
                    image=image,
                    label=item["label"],
                    target_offset=item["target_offset"],
                    loc_mask=item["loc_mask"],
                    candidate_center=item["candidate_center"],
                    target_center=item["target_center"],
                    spacing=item["spacing"],
                    diameter_mm=item["diameter_mm"],
                )
                items.append(
                    {
                        "path": str(file_path),
                        "seriesuid": sample.seriesuid,
                        "label": sample.label,
                        "diameter_mm": sample.diameter_mm,
                    }
                )
                pbar.update(1)
        pbar.close()

        manifest_path.write_text(json.dumps({"config": config, "items": items}, indent=2))
        cached_by_split[split] = items
        print(f"Cached {split} patches: {len(items)} -> {split_dir}")

    return cached_by_split


def make_loader(
    samples: list[PatchSample],
    patch_size_zyx: tuple[int, int, int],
    resample_spacing_xyz: tuple[float, float, float] | None,
    cache_size: int,
    train: bool,
    positive_jitter_mm: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    cached_items: list[dict[str, str | int | float]] | None = None,
    cache_in_memory: bool = False,
) -> DataLoader:
    if cached_items is not None:
        dataset = CachedLuna3DPatchDataset(
            items=cached_items,
            patch_size_zyx=patch_size_zyx,
            train=train,
            seed=seed,
            cache_in_memory=cache_in_memory,
        )
    else:
        dataset = Luna3DPatchDataset(
            samples=samples,
            patch_size_zyx=patch_size_zyx,
            resample_spacing_xyz=resample_spacing_xyz,
            cache_size=cache_size,
            train=train,
            positive_jitter_mm=positive_jitter_mm,
            seed=seed,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def multitask_loss(
    cls_logit: torch.Tensor,
    loc_offset: torch.Tensor,
    labels: torch.Tensor,
    target_offset: torch.Tensor,
    loc_mask: torch.Tensor,
    cls_criterion: nn.Module,
    loc_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_loss = cls_criterion(cls_logit, labels)
    per_axis_loc = nn.functional.smooth_l1_loss(loc_offset, target_offset, reduction="none")
    per_sample_loc = per_axis_loc.mean(dim=1)
    if loc_mask.sum() > 0:
        loc_loss = (per_sample_loc * loc_mask).sum() / loc_mask.sum()
    else:
        loc_loss = per_sample_loc.sum() * 0.0
    return cls_loss + loc_weight * loc_loss, cls_loss.detach(), loc_loss.detach()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    cls_criterion: nn.Module,
    loc_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_loc = 0.0
    y_true: list[int] = []
    y_prob: list[float] = []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=torch.cuda.is_available())
        labels = batch["label"].to(device, non_blocking=torch.cuda.is_available())
        target_offset = batch["target_offset"].to(device, non_blocking=torch.cuda.is_available())
        loc_mask = batch["loc_mask"].to(device, non_blocking=torch.cuda.is_available())

        optimizer.zero_grad(set_to_none=True)
        cls_logit, loc_offset = model(images)
        loss, cls_loss, loc_loss = multitask_loss(
            cls_logit,
            loc_offset,
            labels,
            target_offset,
            loc_mask,
            cls_criterion,
            loc_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_cls += float(cls_loss.item())
        total_loc += float(loc_loss.item())
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(torch.sigmoid(cls_logit).detach().cpu().numpy().tolist())

    return summarize_epoch(total_loss, total_cls, total_loc, y_true, y_prob, len(loader))


def patch_position_from_offset_xyz(
    offset_xyz: np.ndarray,
    patch_size_zyx: tuple[int, int, int],
) -> np.ndarray:
    patch_size_xyz = np.array(
        [patch_size_zyx[2], patch_size_zyx[1], patch_size_zyx[0]],
        dtype=np.float32,
    )
    position_xyz = patch_size_xyz / 2.0 + offset_xyz.astype(np.float32) * (patch_size_xyz / 2.0)
    return np.clip(position_xyz, 0.0, patch_size_xyz - 1.0)


def safe_filename(text: str, max_len: int = 96) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return safe[:max_len].strip("_") or "series"


def save_nodule_prediction_image(
    patch_zyx: np.ndarray,
    seriesuid: str,
    row_index: int,
    probability: float,
    pred_label: int,
    diameter_mm: float,
    loc_error_mm: float,
    target_offset_xyz: np.ndarray,
    pred_offset_xyz: np.ndarray,
    output_dir: Path,
    patch_size_zyx: tuple[int, int, int],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    target_xyz = patch_position_from_offset_xyz(target_offset_xyz, patch_size_zyx)
    pred_xyz = patch_position_from_offset_xyz(pred_offset_xyz, patch_size_zyx)
    target_x, target_y, target_z = target_xyz
    pred_x, pred_y, pred_z = pred_xyz

    slice_z = int(round(target_z))
    slice_y = int(round(target_y))
    slice_x = int(round(target_x))

    views = [
        ("Axial", patch_zyx[slice_z, :, :], (target_x, target_y), (pred_x, pred_y), "x", "y"),
        ("Coronal", patch_zyx[:, slice_y, :], (target_x, target_z), (pred_x, pred_z), "x", "z"),
        ("Sagittal", patch_zyx[:, :, slice_x], (target_y, target_z), (pred_y, pred_z), "y", "z"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    verdict = "Nodule" if pred_label == 1 else "Non-nodule"
    fig.suptitle(
        f"{seriesuid}\n"
        f"prob={probability:.3f} pred={verdict} diameter={diameter_mm:.2f}mm "
        f"loc_error={loc_error_mm:.2f}mm",
        fontsize=9,
    )
    for ax, (title, image, target_xy, pred_xy, xlabel, ylabel) in zip(axes, views):
        ax.imshow(image, cmap="gray", vmin=-1.0, vmax=1.0)
        ax.scatter([target_xy[0]], [target_xy[1]], marker="x", c="lime", s=90, linewidths=2, label="target")
        ax.scatter([pred_xy[0]], [pred_xy[1]], marker="+", c="red", s=110, linewidths=2, label="pred")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.83))

    image_path = output_dir / f"nodule_{row_index:04d}_{safe_filename(seriesuid)}.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    return image_path


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cls_criterion: nn.Module,
    loc_weight: float,
    output_csv: Path | None = None,
    threshold: float = 0.5,
    prediction_image_dir: Path | None = None,
    max_prediction_images: int = 128,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_loc = 0.0
    y_true: list[int] = []
    y_prob: list[float] = []
    loc_errors: list[float] = []
    prediction_rows: list[dict[str, float | int | str]] = []
    saved_prediction_images = 0

    if prediction_image_dir is not None:
        prediction_image_dir.mkdir(parents=True, exist_ok=True)
        for old_image in prediction_image_dir.glob("nodule_*.png"):
            old_image.unlink()

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device, non_blocking=torch.cuda.is_available())
        labels = batch["label"].to(device, non_blocking=torch.cuda.is_available())
        target_offset = batch["target_offset"].to(device, non_blocking=torch.cuda.is_available())
        loc_mask = batch["loc_mask"].to(device, non_blocking=torch.cuda.is_available())

        cls_logit, loc_offset = model(images)
        loss, cls_loss, loc_loss = multitask_loss(
            cls_logit,
            loc_offset,
            labels,
            target_offset,
            loc_mask,
            cls_criterion,
            loc_weight,
        )

        total_loss += float(loss.item())
        total_cls += float(cls_loss.item())
        total_loc += float(loc_loss.item())

        probs = torch.sigmoid(cls_logit).detach().cpu().numpy()
        loc_pred = loc_offset.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(int)
        candidates_np = batch["candidate_center"].detach().cpu().numpy()
        targets_np = batch["target_center"].detach().cpu().numpy()
        spacing_np = batch["spacing"].detach().cpu().numpy()
        target_offsets_np = batch["target_offset"].detach().cpu().numpy()
        diameters_np = batch["diameter_mm"].detach().cpu().numpy()
        images_np = batch["image"].detach().cpu().numpy()

        patch_size_xyz = np.array(
            [loader.dataset.patch_size_zyx[2], loader.dataset.patch_size_zyx[1], loader.dataset.patch_size_zyx[0]],
            dtype=np.float32,
        )
        pred_centers_np = candidates_np + loc_pred * (patch_size_xyz[None, :] / 2.0) * spacing_np

        y_true.extend(labels_np.tolist())
        y_prob.extend(probs.tolist())

        seriesuids = batch["seriesuid"]
        for i, label in enumerate(labels_np):
            row_index = len(prediction_rows)
            loc_error = float("nan")
            if label == 1:
                loc_error = float(np.linalg.norm(pred_centers_np[i] - targets_np[i]))
                loc_errors.append(loc_error)

            visualization_path = ""
            if (
                prediction_image_dir is not None
                and label == 1
                and saved_prediction_images < max_prediction_images
            ):
                visualization_path = str(
                    save_nodule_prediction_image(
                        patch_zyx=images_np[i, 0],
                        seriesuid=str(seriesuids[i]),
                        row_index=row_index,
                        probability=float(probs[i]),
                        pred_label=int(probs[i] >= threshold),
                        diameter_mm=float(diameters_np[i]),
                        loc_error_mm=loc_error,
                        target_offset_xyz=target_offsets_np[i],
                        pred_offset_xyz=loc_pred[i],
                        output_dir=prediction_image_dir,
                        patch_size_zyx=loader.dataset.patch_size_zyx,
                    )
                )
                saved_prediction_images += 1

            if output_csv is not None:
                prediction_rows.append(
                    {
                        "seriesuid": str(seriesuids[i]),
                        "label": int(label),
                        "prob_nodule": float(probs[i]),
                        "pred_label": int(probs[i] >= threshold),
                        "diameter_mm": float(diameters_np[i]),
                        "candidate_x": float(candidates_np[i, 0]),
                        "candidate_y": float(candidates_np[i, 1]),
                        "candidate_z": float(candidates_np[i, 2]),
                        "target_x": float(targets_np[i, 0]),
                        "target_y": float(targets_np[i, 1]),
                        "target_z": float(targets_np[i, 2]),
                        "pred_x": float(pred_centers_np[i, 0]),
                        "pred_y": float(pred_centers_np[i, 1]),
                        "pred_z": float(pred_centers_np[i, 2]),
                        "target_offset_x": float(target_offsets_np[i, 0]),
                        "target_offset_y": float(target_offsets_np[i, 1]),
                        "target_offset_z": float(target_offsets_np[i, 2]),
                        "pred_offset_x": float(loc_pred[i, 0]),
                        "pred_offset_y": float(loc_pred[i, 1]),
                        "pred_offset_z": float(loc_pred[i, 2]),
                        "loc_error_mm": loc_error,
                        "visualization_path": visualization_path,
                    }
                )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()) if prediction_rows else [])
            if prediction_rows:
                writer.writeheader()
                writer.writerows(prediction_rows)
        nodule_rows = [row for row in prediction_rows if int(row["label"]) == 1]
        nodule_csv = output_csv.parent / "test_nodule_predictions_3d.csv"
        with nodule_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()) if prediction_rows else [])
            if nodule_rows:
                writer.writeheader()
                writer.writerows(nodule_rows)

    metrics = summarize_epoch(total_loss, total_cls, total_loc, y_true, y_prob, len(loader), threshold)
    metrics["loc_error_mean_mm"] = float(np.mean(loc_errors)) if loc_errors else float("nan")
    metrics["loc_error_median_mm"] = float(np.median(loc_errors)) if loc_errors else float("nan")
    metrics["saved_prediction_images"] = float(saved_prediction_images)
    return metrics


def summarize_epoch(
    total_loss: float,
    total_cls: float,
    total_loc: float,
    y_true: list[int],
    y_prob: list[float],
    num_batches: int,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = [int(p >= threshold) for p in y_prob]
    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "cls_loss": total_cls / max(num_batches, 1),
        "loc_loss": total_loc / max(num_batches, 1),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0) if y_true else float("nan"),
        "recall": recall_score(y_true, y_pred, zero_division=0) if y_true else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0) if y_true else float("nan"),
        "auc": float("nan"),
    }
    if len(set(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history_3d.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def load_history(output_dir: Path) -> list[dict[str, float]]:
    history_path = output_dir / "training_history_3d.csv"
    if not history_path.exists():
        return []
    return pd.read_csv(history_path).to_dict("records")


def save_plots(history: list[dict[str, float]], test_predictions_csv: Path, output_dir: Path) -> None:
    if history:
        epochs = [row["epoch"] for row in history]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row["train_loss"] for row in history], label="Train")
        plt.plot(epochs, [row["val_loss"] for row in history], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("3D Multi-task Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve_3d.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row["train_accuracy"] for row in history], label="Train Accuracy")
        plt.plot(epochs, [row["val_accuracy"] for row in history], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Classification Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_curve_3d.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row["train_auc"] for row in history], label="Train AUC")
        plt.plot(epochs, [row["val_auc"] for row in history], label="Validation AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title("Classification AUC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "auc_curve_3d.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row["val_loc_error_mean_mm"] for row in history])
        plt.xlabel("Epoch")
        plt.ylabel("Mean Error (mm)")
        plt.title("Validation Location Error")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "location_error_curve_3d.png", dpi=200)
        plt.close()

    if test_predictions_csv.exists():
        predictions = pd.read_csv(test_predictions_csv)
        if not predictions.empty and predictions["label"].nunique() == 2:
            y_true = predictions["label"].to_numpy()
            y_score = predictions["prob_nodule"].to_numpy()
            y_pred = predictions["pred_label"].to_numpy()

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(5, 4))
            plt.imshow(cm, cmap="Blues")
            plt.xticks([0, 1], ["Negative", "Positive"])
            plt.yticks([0, 1], ["Negative", "Positive"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Test Confusion Matrix")
            for y in range(cm.shape[0]):
                for x in range(cm.shape[1]):
                    plt.text(x, y, str(cm[y, x]), ha="center", va="center")
            plt.tight_layout()
            plt.savefig(output_dir / "confusion_matrix_3d.png", dpi=200)
            plt.close()

            fpr, tpr, _ = roc_curve(y_true, y_score)
            plt.figure(figsize=(8, 5))
            plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Test ROC Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "roc_curve_3d.png", dpi=200)
            plt.close()

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.figure(figsize=(8, 5))
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Test Precision-Recall Curve")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "pr_curve_3d.png", dpi=200)
            plt.close()


def class_pos_weight(samples: list[PatchSample], device: torch.device) -> torch.Tensor:
    positives = sum(1 for sample in samples if sample.label == 1)
    negatives = sum(1 for sample in samples if sample.label == 0)
    if positives == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(max(1.0, negatives / positives), dtype=torch.float32, device=device)


def save_split_manifest(samples_by_split: dict[str, list[PatchSample]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        split: [asdict(sample) for sample in samples]
        for split, samples in samples_by_split.items()
    }
    (output_dir / "split_samples_3d.json").write_text(json.dumps(manifest, indent=2))


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size_zyx = parse_int_tuple(args.patch_size, 3, "patch-size")
    resample_spacing_xyz = None if args.no_resample else parse_float_tuple(args.resample_spacing, 3, "resample-spacing")

    split_subsets = {
        "train": tuple(args.train_subsets),
        "val": tuple(args.val_subsets),
        "test": tuple(args.test_subsets),
    }
    samples_by_split = collect_samples(
        data_dir=Path(args.data_dir),
        split_subsets=split_subsets,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        resample_spacing_xyz=resample_spacing_xyz,
    )
    save_split_manifest(samples_by_split, output_dir)

    if args.dry_run:
        print("Dry run completed. Sample manifest saved; training was skipped.")
        return

    cached_by_split = None
    if args.precache_patches:
        cached_by_split = build_or_load_patch_cache(
            samples_by_split=samples_by_split,
            patch_size_zyx=patch_size_zyx,
            resample_spacing_xyz=resample_spacing_xyz,
            cache_dir=Path(args.patch_cache_dir),
            cache_size=args.cache_size,
            positive_jitter_mm=args.positive_jitter_mm,
            train_cache_copies=args.train_cache_copies,
            seed=args.seed,
            rebuild=args.rebuild_patch_cache,
            cache_float16=args.cache_float16,
        )

    train_loader = make_loader(
        samples_by_split["train"],
        patch_size_zyx,
        resample_spacing_xyz,
        args.cache_size,
        train=True,
        positive_jitter_mm=args.positive_jitter_mm,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cached_items=None if cached_by_split is None else cached_by_split["train"],
        cache_in_memory=args.cache_in_memory,
    )
    val_loader = make_loader(
        samples_by_split["val"],
        patch_size_zyx,
        resample_spacing_xyz,
        args.cache_size,
        train=False,
        positive_jitter_mm=0.0,
        seed=args.seed + 1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cached_items=None if cached_by_split is None else cached_by_split["val"],
        cache_in_memory=args.cache_in_memory,
    )
    test_loader = make_loader(
        samples_by_split["test"],
        patch_size_zyx,
        resample_spacing_xyz,
        args.cache_size,
        train=False,
        positive_jitter_mm=0.0,
        seed=args.seed + 2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cached_items=None if cached_by_split is None else cached_by_split["test"],
        cache_in_memory=args.cache_in_memory,
    )

    device = get_device()
    print("Using device:", device)

    model = Light3DCNNTransformer(
        patch_size_zyx=patch_size_zyx,
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
    ).to(device)

    pos_weight = class_pos_weight(samples_by_split["train"], device)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path is not None else output_dir / "best_light3d_cnn_transformer_luna16.pth"
    prediction_image_dir = (
        Path(args.prediction_image_dir)
        if args.prediction_image_dir is not None
        else output_dir / "test_nodule_visualizations"
    )
    if not args.save_prediction_images:
        prediction_image_dir = None

    if args.eval_only:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_predictions_csv = output_dir / "test_predictions_3d.csv"
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            cls_criterion,
            args.loc_weight,
            output_csv=test_predictions_csv,
            prediction_image_dir=prediction_image_dir,
            max_prediction_images=args.max_prediction_images,
        )

        (output_dir / "test_metrics_3d.json").write_text(json.dumps(test_metrics, indent=2))
        save_plots(load_history(output_dir), test_predictions_csv, output_dir)

        print("Loaded checkpoint:", checkpoint_path)
        print("Test predictions:", test_predictions_csv)
        if prediction_image_dir is not None:
            print("Nodule prediction images:", prediction_image_dir)
        print("Test metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        return

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    best_score = -float("inf")
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cls_criterion,
            args.loc_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            cls_criterion,
            args.loc_weight,
        )

        score = val_metrics["auc"]
        if math.isnan(score):
            score = val_metrics["f1"]
        scheduler.step(score)

        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "val_loc_error_mean_mm": val_metrics["loc_error_mean_mm"],
            "val_loc_error_median_mm": val_metrics["loc_error_median_mm"],
        }
        history.append(row)
        save_history(history, output_dir)

        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            wait = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "patch_size_zyx": patch_size_zyx,
                    "resample_spacing_xyz": resample_spacing_xyz,
                    "args": vars(args),
                    "best_score": best_score,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            wait += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"train_auc={train_metrics['auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_loc_mean={val_metrics['loc_error_mean_mm']:.2f}mm "
            f"patience={wait}/{args.patience}"
        )

        if not args.disable_early_stopping and wait >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_predictions_csv = output_dir / "test_predictions_3d.csv"
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        cls_criterion,
        args.loc_weight,
        output_csv=test_predictions_csv,
        prediction_image_dir=prediction_image_dir,
        max_prediction_images=args.max_prediction_images,
    )

    (output_dir / "test_metrics_3d.json").write_text(json.dumps(test_metrics, indent=2))
    save_plots(history, test_predictions_csv, output_dir)

    print("Best checkpoint:", checkpoint_path)
    print("Test predictions:", test_predictions_csv)
    if prediction_image_dir is not None:
        print("Nodule prediction images:", prediction_image_dir)
    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ultra-fast lightweight 3D CNN + Transformer on LUNA16.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-subsets", type=int, nargs="+", default=TRAIN_SUBSETS)
    parser.add_argument("--val-subsets", type=int, nargs="+", default=VAL_SUBSETS)
    parser.add_argument("--test-subsets", type=int, nargs="+", default=TEST_SUBSETS)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(24, 24, 24), metavar=("D", "H", "W"))
    parser.add_argument("--resample-spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("SX", "SY", "SZ"))
    parser.add_argument("--no-resample", dest="no_resample", action="store_true", default=True, help="Use original CT voxel spacing instead of 1 mm isotropic spacing.")
    parser.add_argument("--resample", dest="no_resample", action="store_false", help="Enable 1 mm isotropic resampling. Slower, but more physically consistent.")
    parser.add_argument("--negative-ratio", type=float, default=0.5, help="Number of sampled negatives per positive annotation.")
    parser.add_argument("--positive-jitter-mm", type=float, default=2.0, help="Training-time random crop-center jitter for positives.")
    parser.add_argument("--cache-size", type=int, default=2, help="Number of CT volumes cached per DataLoader worker.")
    parser.add_argument("--precache-patches", dest="precache_patches", action="store_true", default=True, help="Precompute cropped 3D patches to disk and train from the patch cache.")
    parser.add_argument("--no-precache-patches", dest="precache_patches", action="store_false", help="Disable patch pre-caching. Much slower for full CT training.")
    parser.add_argument("--patch-cache-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "patch_cache_ultrafast")
    parser.add_argument("--rebuild-patch-cache", action="store_true", help="Delete and rebuild cached patch files for the current settings.")
    parser.add_argument("--train-cache-copies", type=int, default=1, help="Cached crop variants per training sample. Keep 1 for maximum speed.")
    parser.add_argument("--cache-float16", dest="cache_float16", action="store_true", default=True, help="Store cached patch intensities as float16 to reduce disk I/O.")
    parser.add_argument("--cache-float32", dest="cache_float16", action="store_false", help="Store cached patch intensities as float32.")
    parser.add_argument("--cache-in-memory", dest="cache_in_memory", action="store_true", default=True, help="Load cached patches into RAM before training for maximum epoch speed.")
    parser.add_argument("--no-cache-in-memory", dest="cache_in_memory", action="store_false", help="Read cached patches from disk every batch.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--disable-early-stopping", dest="disable_early_stopping", action="store_true", default=True, help="Always run all requested epochs.")
    parser.add_argument("--enable-early-stopping", dest="disable_early_stopping", action="store_false", help="Stop early when validation score stops improving.")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loc-weight", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="Build split/sample manifest without training.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate the checkpoint on the test split.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Checkpoint to load for --eval-only or final testing.")
    parser.add_argument("--save-prediction-images", dest="save_prediction_images", action="store_true", default=True, help="Save positive test nodule visualizations with predicted and target centers.")
    parser.add_argument("--no-save-prediction-images", dest="save_prediction_images", action="store_false", help="Do not save nodule visualization PNGs.")
    parser.add_argument("--prediction-image-dir", type=Path, default=None, help="Directory for positive test nodule visualization PNGs.")
    parser.add_argument("--max-prediction-images", type=int, default=128, help="Maximum number of positive test nodule PNGs to save.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

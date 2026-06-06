"""
bpe_lm_fusion/data.py
Load ASR dataset splits. See docs/dataset_description.md.

KenLM corpus는 train(+옵션 test) TEXT만 사용. 평가는 test audio+text.
audio는 raw float 리스트이므로 np.float32 변환 후 feature extractor에 투입.
"""
from __future__ import annotations
import numpy as np


def load_dataset(path: str):
    from datasets import load_from_disk
    return load_from_disk(path)


def split_texts(ds, split: str) -> list[str]:
    """Return list of raw transcript strings for a split."""
    return list(ds[split]["text"])


def audio_to_array(row) -> np.ndarray:
    return np.asarray(row["audio"], dtype=np.float32)


def audio_duration_seconds(row) -> float:
    """Return audio duration from row audio length and sampling rate."""
    sr = int(row["sampling_rate"])
    return len(row["audio"]) / sr if sr else 0.0

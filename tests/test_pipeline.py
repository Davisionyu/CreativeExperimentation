from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diabetes_prediction.config import DEFAULT_CONFIG
from diabetes_prediction.data import split_features_target
from diabetes_prediction.features import build_preprocessor
from diabetes_prediction.logging_utils import setup_logging


def test_split_features_target_maps_binary_labels() -> None:
    df = pd.DataFrame(
        {
            "Age": [40, 55],
            "Gender": ["Male", "Female"],
            "class": ["Positive", "Negative"],
        }
    )
    X, y = split_features_target(df, DEFAULT_CONFIG)
    assert list(X.columns) == ["Age", "Gender"]
    assert y.tolist() == [1, 0]


def test_preprocessor_transforms_mixed_features(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path, "test")
    X = pd.DataFrame({"Age": [40, 55], "Gender": ["Male", "Female"], "Polyuria": ["Yes", "No"]})
    preprocessor = build_preprocessor(X, logger)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= 2

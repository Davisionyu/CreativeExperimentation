"""Data loading, validation, and split helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_prediction.config import ProjectConfig


EXPECTED_COLUMNS = [
    "Age",
    "Gender",
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
    "class",
]


def load_dataset(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load the CSV dataset using common encodings."""

    try:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                df = pd.read_csv(path, encoding=encoding)
                logger.info("Loaded dataset from %s with encoding=%s, shape=%s", path, encoding, df.shape)
                return df
            except UnicodeDecodeError as exc:
                last_error = exc
                logger.warning("Failed reading %s with encoding=%s", path, encoding)
        raise ValueError(f"Unable to read dataset with supported encodings: {path}") from last_error
    except Exception:
        logger.exception("Dataset loading failed")
        raise


def validate_dataset(df: pd.DataFrame, config: ProjectConfig, logger: logging.Logger) -> None:
    """Validate schema and label quality before training."""

    try:
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        if df.empty:
            raise ValueError("Dataset is empty")
        if df[config.target_column].nunique() != 2:
            raise ValueError("Target column must be binary")
        if df.isna().any().any():
            logger.warning("Dataset contains missing values; preprocessing will impute them")

        duplicated = int(df.duplicated().sum())
        if duplicated:
            logger.info("Detected %s duplicated rows; keeping them because the source dataset is small", duplicated)
        logger.info("Dataset validation passed")
    except Exception:
        logger.exception("Dataset validation failed")
        raise


def normalize_binary_text(df: pd.DataFrame, columns: Iterable[str], logger: logging.Logger) -> pd.DataFrame:
    """Normalize common categorical spelling differences."""

    try:
        normalized = df.copy()
        for column in columns:
            if column in normalized.columns and normalized[column].dtype == "object":
                normalized[column] = normalized[column].astype(str).str.strip()
        logger.info("Normalized text columns: %s", list(columns))
        return normalized
    except Exception:
        logger.exception("Categorical normalization failed")
        raise


def split_features_target(df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the binary target."""

    y = df[config.target_column].map({config.positive_label: 1, "Negative": 0})
    if y.isna().any():
        unexpected = sorted(df.loc[y.isna(), config.target_column].astype(str).unique())
        raise ValueError(f"Unexpected target labels: {unexpected}")
    X = df.drop(columns=[config.target_column])
    return X, y.astype(int)


def train_validation_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train, validation, and test splits."""

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    relative_validation = config.validation_size / (1.0 - config.test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=relative_validation,
        stratify=y_train_full,
        random_state=config.random_state,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

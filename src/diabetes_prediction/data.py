"""数据读取、校验和划分工具。"""

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
    """使用常见编码读取 CSV 数据集。"""

    try:
        if not path.exists():
            raise FileNotFoundError(f"未找到数据集：{path}")

        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                df = pd.read_csv(path, encoding=encoding)
                logger.info("已读取数据集：%s，编码=%s，形状=%s", path, encoding, df.shape)
                return df
            except UnicodeDecodeError as exc:
                last_error = exc
                logger.warning("使用编码 %s 读取失败：%s", encoding, path)
        raise ValueError(f"无法使用支持的编码读取数据集：{path}") from last_error
    except Exception:
        logger.exception("数据集读取失败")
        raise


def validate_dataset(df: pd.DataFrame, config: ProjectConfig, logger: logging.Logger) -> None:
    """在训练前校验字段结构和标签质量。"""

    try:
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要字段：{missing_columns}")
        if df.empty:
            raise ValueError("数据集为空")
        if df[config.target_column].nunique() != 2:
            raise ValueError("目标列必须是二分类")
        if df.isna().any().any():
            logger.warning("数据集中存在缺失值，预处理阶段会自动填补")

        duplicated = int(df.duplicated().sum())
        if duplicated:
            logger.info("检测到 %s 条重复样本；由于原始数据集较小，保留这些样本", duplicated)
        logger.info("数据集校验通过")
    except Exception:
        logger.exception("数据集校验失败")
        raise


def normalize_binary_text(df: pd.DataFrame, columns: Iterable[str], logger: logging.Logger) -> pd.DataFrame:
    """统一常见类别字段的空格和格式差异。"""

    try:
        normalized = df.copy()
        for column in columns:
            if column in normalized.columns and normalized[column].dtype == "object":
                normalized[column] = normalized[column].astype(str).str.strip()
        logger.info("已统一文本字段格式：%s", list(columns))
        return normalized
    except Exception:
        logger.exception("类别字段规范化失败")
        raise


def split_features_target(df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.Series]:
    """分离特征和二分类目标。"""

    y = df[config.target_column].map({config.positive_label: 1, "Negative": 0})
    if y.isna().any():
        unexpected = sorted(df.loc[y.isna(), config.target_column].astype(str).unique())
        raise ValueError(f"发现未预期的目标标签：{unexpected}")
    X = df.drop(columns=[config.target_column])
    return X, y.astype(int)


def train_validation_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """创建分层训练集、验证集和测试集。"""

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

"""特征预处理流程。"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """推断数值特征列和类别特征列。"""

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame, logger: logging.Logger) -> ColumnTransformer:
    """构建适用于数值与类别混合健康指标的预处理器。"""

    try:
        numeric_features, categorical_features = infer_feature_columns(X)
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
            ]
        )
        logger.info("数值特征=%s；类别特征=%s", numeric_features, categorical_features)
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )
    except Exception:
        logger.exception("预处理器构建失败")
        raise


def build_feature_selector(k: int) -> SelectKBest:
    """选择最有信息量的特征。"""

    return SelectKBest(score_func=mutual_info_classif, k=k)

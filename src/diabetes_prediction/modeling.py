"""训练、评估和模型产物工具。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from diabetes_prediction.config import ProjectConfig
from diabetes_prediction.features import build_feature_selector, build_preprocessor


def build_candidate_models(X: pd.DataFrame, config: ProjectConfig, logger: logging.Logger) -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    """创建候选模型管线和小型参数网格。"""

    try:
        transformed_feature_count = len(X.columns)
        k = min(config.feature_k, transformed_feature_count)
        preprocessor = build_preprocessor(X, logger)
        candidates: dict[str, tuple[Pipeline, dict[str, Any]]] = {
            "logistic_regression": (
                Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        ("select", build_feature_selector(k)),
                        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)),
                    ]
                ),
                {"model__C": [0.1, 1.0, 5.0]},
            ),
            "random_forest": (
                Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        ("select", build_feature_selector(k)),
                        ("model", RandomForestClassifier(class_weight="balanced", random_state=config.random_state)),
                    ]
                ),
                {
                    "model__n_estimators": [120],
                    "model__max_depth": [8, 12],
                    "model__min_samples_leaf": [2],
                },
            ),
            "gradient_boosting": (
                Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        ("select", build_feature_selector(k)),
                        ("model", GradientBoostingClassifier(random_state=config.random_state)),
                    ]
                ),
                {
                    "model__n_estimators": [80, 120],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [2],
                },
            ),
        }
        logger.info("已构建 %s 个候选模型管线", len(candidates))
        return candidates
    except Exception:
        logger.exception("候选模型构建失败")
        raise


def tune_models(
    candidates: dict[str, tuple[Pipeline, dict[str, Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ProjectConfig,
    logger: logging.Logger,
) -> tuple[str, GridSearchCV, list[dict[str, Any]]]:
    """使用分层交叉验证调参，并按 F1 选择最优模型。"""

    try:
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        summaries: list[dict[str, Any]] = []
        best_name = ""
        best_search: GridSearchCV | None = None
        best_score = -np.inf

        for name, (pipeline, param_grid) in candidates.items():
            logger.info("正在调参模型=%s，参数网格=%s", name, param_grid)
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            summary = {
                "模型名称": name,
                "最佳交叉验证F1": float(search.best_score_),
                "最佳参数": search.best_params_,
            }
            summaries.append(summary)
            logger.info("模型=%s 的最佳交叉验证 F1=%.4f，参数=%s", name, search.best_score_, search.best_params_)
            if search.best_score_ > best_score:
                best_name = name
                best_search = search
                best_score = float(search.best_score_)

        if best_search is None:
            raise RuntimeError("没有成功调优出可用模型")
        logger.info("已选择最优模型=%s，交叉验证 F1=%.4f", best_name, best_score)
        return best_name, best_search, summaries
    except Exception:
        logger.exception("模型调参失败")
        raise


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, split_name: str, logger: logging.Logger) -> dict[str, Any]:
    """评估已训练模型在某个数据划分上的表现。"""

    try:
        y_pred = model.predict(X)
        y_score = model.predict_proba(X)[:, 1]
        metrics = {
            "数据划分": split_name,
            "准确率": float(accuracy_score(y, y_pred)),
            "精确率": float(precision_score(y, y_pred, zero_division=0)),
            "召回率": float(recall_score(y, y_pred, zero_division=0)),
            "F1分数": float(f1_score(y, y_pred, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y, y_score)),
            "混淆矩阵": confusion_matrix(y, y_pred).tolist(),
        }
        logger.info("评估 %s 指标=%s", split_name, metrics)
        return metrics
    except Exception:
        logger.exception("模型评估失败，划分=%s", split_name)
        raise


def collect_badcases(model: Pipeline, X: pd.DataFrame, y: pd.Series, logger: logging.Logger) -> pd.DataFrame:
    """收集误分类样本供后续分析。"""

    try:
        y_pred = model.predict(X)
        y_score = model.predict_proba(X)[:, 1]
        badcases = X.loc[y_pred != y.to_numpy()].copy()
        badcases["实际标签"] = y.loc[badcases.index].map({1: "阳性", 0: "阴性"})
        badcases["预测标签"] = pd.Series(y_pred, index=X.index).loc[badcases.index].map({1: "阳性", 0: "阴性"})
        badcases["阳性概率"] = pd.Series(y_score, index=X.index).loc[badcases.index].round(4)
        logger.info("已收集 %s 条误判样本", len(badcases))
        return badcases
    except Exception:
        logger.exception("误判样本收集失败")
        raise


def extract_feature_report(model: Pipeline, logger: logging.Logger) -> pd.DataFrame:
    """导出被选中特征得分和模型重要性。"""

    try:
        preprocess = model.named_steps["preprocess"]
        selector = model.named_steps["select"]
        estimator = model.named_steps["model"]
        names = preprocess.get_feature_names_out()
        selected_mask = selector.get_support()
        selected_names = names[selected_mask]
        scores = selector.scores_[selected_mask]

        report = pd.DataFrame({"特征": selected_names, "特征选择分数": scores})
        if hasattr(estimator, "feature_importances_"):
            report["模型重要性"] = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            report["模型重要性"] = np.abs(estimator.coef_[0])
        else:
            report["模型重要性"] = np.nan
        report = report.sort_values(["模型重要性", "特征选择分数"], ascending=False)
        logger.info("已导出特征分析结果，共 %s 行", len(report))
        return report
    except Exception:
        logger.exception("特征分析结果导出失败")
        raise


def save_json(data: Any, path: Path, logger: logging.Logger) -> None:
    """以 UTF-8 编码保存 JSON。"""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("已保存 JSON 产物：%s", path)
    except Exception:
        logger.exception("保存 JSON 失败：%s", path)
        raise


def save_model(model: Pipeline, path: Path, logger: logging.Logger) -> None:
    """保存已训练的 sklearn 管线。"""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("已保存模型产物：%s", path)
    except Exception:
        logger.exception("保存模型失败：%s", path)
        raise


def load_model(path: Path, logger: logging.Logger) -> Pipeline:
    """读取已训练的 sklearn 管线。"""

    try:
        if not path.exists():
            raise FileNotFoundError(f"未找到模型产物：{path}")
        model = joblib.load(path)
        logger.info("已读取模型产物：%s", path)
        return model
    except Exception:
        logger.exception("读取模型失败：%s", path)
        raise

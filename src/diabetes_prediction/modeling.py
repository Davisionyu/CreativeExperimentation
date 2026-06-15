"""Training, evaluation, and model artifact helpers."""

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
    classification_report,
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
    """Create candidate model pipelines and small parameter grids."""

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
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 6, 10],
                    "model__min_samples_leaf": [1, 2],
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
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [2, 3],
                },
            ),
        }
        logger.info("Built %s candidate model pipelines", len(candidates))
        return candidates
    except Exception:
        logger.exception("Candidate model construction failed")
        raise


def tune_models(
    candidates: dict[str, tuple[Pipeline, dict[str, Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ProjectConfig,
    logger: logging.Logger,
) -> tuple[str, GridSearchCV, list[dict[str, Any]]]:
    """Tune candidates with stratified cross validation and select the best by F1."""

    try:
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        summaries: list[dict[str, Any]] = []
        best_name = ""
        best_search: GridSearchCV | None = None
        best_score = -np.inf

        for name, (pipeline, param_grid) in candidates.items():
            logger.info("Tuning model=%s with grid=%s", name, param_grid)
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
                "model": name,
                "best_cv_f1": float(search.best_score_),
                "best_params": search.best_params_,
            }
            summaries.append(summary)
            logger.info("Model=%s best_cv_f1=%.4f params=%s", name, search.best_score_, search.best_params_)
            if search.best_score_ > best_score:
                best_name = name
                best_search = search
                best_score = float(search.best_score_)

        if best_search is None:
            raise RuntimeError("No model was successfully tuned")
        logger.info("Selected best model=%s with cv_f1=%.4f", best_name, best_score)
        return best_name, best_search, summaries
    except Exception:
        logger.exception("Model tuning failed")
        raise


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, split_name: str, logger: logging.Logger) -> dict[str, Any]:
    """Evaluate a fitted model on one data split."""

    try:
        y_pred = model.predict(X)
        y_score = model.predict_proba(X)[:, 1]
        metrics = {
            "split": split_name,
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_score)),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, target_names=["Negative", "Positive"], zero_division=0),
        }
        logger.info("Evaluation %s metrics=%s", split_name, {k: v for k, v in metrics.items() if k != "classification_report"})
        return metrics
    except Exception:
        logger.exception("Model evaluation failed for split=%s", split_name)
        raise


def collect_badcases(model: Pipeline, X: pd.DataFrame, y: pd.Series, logger: logging.Logger) -> pd.DataFrame:
    """Collect misclassified test rows for later analysis."""

    try:
        y_pred = model.predict(X)
        y_score = model.predict_proba(X)[:, 1]
        badcases = X.loc[y_pred != y.to_numpy()].copy()
        badcases["actual"] = y.loc[badcases.index].map({1: "Positive", 0: "Negative"})
        badcases["predicted"] = pd.Series(y_pred, index=X.index).loc[badcases.index].map({1: "Positive", 0: "Negative"})
        badcases["positive_probability"] = pd.Series(y_score, index=X.index).loc[badcases.index].round(4)
        logger.info("Collected %s badcases", len(badcases))
        return badcases
    except Exception:
        logger.exception("Badcase collection failed")
        raise


def extract_feature_report(model: Pipeline, logger: logging.Logger) -> pd.DataFrame:
    """Export selected feature scores and model importances when available."""

    try:
        preprocess = model.named_steps["preprocess"]
        selector = model.named_steps["select"]
        estimator = model.named_steps["model"]
        names = preprocess.get_feature_names_out()
        selected_mask = selector.get_support()
        selected_names = names[selected_mask]
        scores = selector.scores_[selected_mask]

        report = pd.DataFrame({"feature": selected_names, "selection_score": scores})
        if hasattr(estimator, "feature_importances_"):
            report["model_importance"] = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            report["model_importance"] = np.abs(estimator.coef_[0])
        else:
            report["model_importance"] = np.nan
        report = report.sort_values(["model_importance", "selection_score"], ascending=False)
        logger.info("Extracted feature report with %s rows", len(report))
        return report
    except Exception:
        logger.exception("Feature report extraction failed")
        raise


def save_json(data: Any, path: Path, logger: logging.Logger) -> None:
    """Persist JSON with UTF-8 encoding."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved JSON artifact: %s", path)
    except Exception:
        logger.exception("Saving JSON failed: %s", path)
        raise


def save_model(model: Pipeline, path: Path, logger: logging.Logger) -> None:
    """Persist a fitted sklearn pipeline."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("Saved model artifact: %s", path)
    except Exception:
        logger.exception("Saving model failed: %s", path)
        raise


def load_model(path: Path, logger: logging.Logger) -> Pipeline:
    """Load a fitted sklearn pipeline."""

    try:
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        model = joblib.load(path)
        logger.info("Loaded model artifact: %s", path)
        return model
    except Exception:
        logger.exception("Loading model failed: %s", path)
        raise

"""Train and evaluate diabetes risk prediction models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diabetes_prediction.config import DEFAULT_CONFIG, ProjectConfig
from diabetes_prediction.data import (
    load_dataset,
    normalize_binary_text,
    split_features_target,
    train_validation_test_split,
    validate_dataset,
)
from diabetes_prediction.logging_utils import setup_logging
from diabetes_prediction.modeling import (
    build_candidate_models,
    collect_badcases,
    evaluate_model,
    extract_feature_report,
    save_json,
    save_model,
    tune_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diabetes risk prediction models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_CONFIG.data_path, help="Path to CSV dataset.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_CONFIG.model_dir, help="Directory for model artifacts.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_CONFIG.report_dir, help="Directory for reports.")
    parser.add_argument("--feature-k", type=int, default=DEFAULT_CONFIG.feature_k, help="Number of selected transformed features.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig(data_path=args.data, model_dir=args.model_dir, report_dir=args.report_dir, feature_k=args.feature_k)
    logger = setup_logging(config.log_dir, "train")
    try:
        df = load_dataset(config.data_path, logger)
        validate_dataset(df, config, logger)
        df = normalize_binary_text(df, df.columns, logger)
        X, y = split_features_target(df, config)
        X_train, X_valid, X_test, y_train, y_valid, y_test = train_validation_test_split(X, y, config)

        candidates = build_candidate_models(X_train, config, logger)
        best_name, best_search, cv_summaries = tune_models(candidates, X_train, y_train, config, logger)
        best_model = best_search.best_estimator_

        validation_metrics = evaluate_model(best_model, X_valid, y_valid, "validation", logger)
        test_metrics = evaluate_model(best_model, X_test, y_test, "test", logger)
        badcases = collect_badcases(best_model, X_test, y_test, logger)
        feature_report = extract_feature_report(best_model, logger)

        config.model_dir.mkdir(parents=True, exist_ok=True)
        config.report_dir.mkdir(parents=True, exist_ok=True)
        save_model(best_model, config.model_dir / "best_model.joblib", logger)
        save_json(
            {
                "best_model": best_name,
                "cv_results": cv_summaries,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "data_shape": list(df.shape),
                "selected_feature_k": config.feature_k,
            },
            config.report_dir / "metrics.json",
            logger,
        )
        badcases.to_csv(config.report_dir / "badcases.csv", index=False, encoding="utf-8-sig")
        feature_report.to_csv(config.report_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
        logger.info("Training completed successfully")
        return 0
    except Exception as exc:
        logger.exception("Training command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

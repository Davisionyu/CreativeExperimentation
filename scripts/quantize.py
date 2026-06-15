"""训练轻量 Logistic 模型并导出 int8 量化权重。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diabetes_prediction.config import DEFAULT_CONFIG, ProjectConfig
from diabetes_prediction.data import load_dataset, normalize_binary_text, split_features_target, train_validation_test_split, validate_dataset
from diabetes_prediction.features import build_feature_selector, build_preprocessor
from diabetes_prediction.logging_utils import setup_logging
from diabetes_prediction.modeling import evaluate_model, save_json, save_model
from diabetes_prediction.quantization import predict_with_quantized_logits, quantize_logistic_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 int8 Logistic 推理模型。")
    parser.add_argument("--data", type=Path, default=DEFAULT_CONFIG.data_path, help="CSV 数据集路径。")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_CONFIG.model_dir / "logistic_model.joblib", help="sklearn Logistic 管线输出路径。")
    parser.add_argument("--quantized-output", type=Path, default=DEFAULT_CONFIG.model_dir / "logistic_int8.json", help="量化模型输出路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ProjectConfig(data_path=args.data)
    logger = setup_logging(config.log_dir, "quantize")
    try:
        df = load_dataset(config.data_path, logger)
        validate_dataset(df, config, logger)
        df = normalize_binary_text(df, df.columns, logger)
        X, y = split_features_target(df, config)
        X_train, _, X_test, y_train, _, y_test = train_validation_test_split(X, y, config)

        model = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train, logger)),
                ("select", build_feature_selector(min(config.feature_k, len(X_train.columns)))),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)),
            ]
        )
        model.fit(X_train, y_train)
        sklearn_metrics = evaluate_model(model, X_test, y_test, "量化源模型测试集", logger)
        artifact = quantize_logistic_pipeline(model, args.quantized_output, logger)

        transformed = model.named_steps["preprocess"].transform(X_test)
        selected = model.named_steps["select"].transform(transformed)
        labels, probabilities = predict_with_quantized_logits(selected, artifact)
        quantized_metrics = {
            "准确率": float((labels == y_test.to_numpy()).mean()),
            "平均阳性概率": float(probabilities.mean()),
        }
        save_model(model, args.model_output, logger)
        save_json(
            {"源模型指标": sklearn_metrics, "量化模型指标": quantized_metrics},
            config.report_dir / "quantization_metrics.json",
            logger,
        )
        logger.info("量化完成")
        return 0
    except Exception as exc:
        logger.exception("量化命令失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

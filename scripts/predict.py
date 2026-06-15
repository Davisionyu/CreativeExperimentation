"""Run inference with the trained diabetes risk prediction model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diabetes_prediction.config import DEFAULT_CONFIG
from diabetes_prediction.logging_utils import setup_logging
from diabetes_prediction.modeling import load_model
from diabetes_prediction.quantization import load_quantized_artifact, predict_with_quantized_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict diabetes risk for CSV rows.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV containing feature columns.")
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"), help="Output CSV path.")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.joblib"), help="Trained sklearn model path.")
    parser.add_argument("--quantized-model", type=Path, default=None, help="Optional int8 logistic artifact path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(DEFAULT_CONFIG.log_dir, "predict")
    try:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        model = load_model(args.model, logger)
        data = pd.read_csv(args.input, encoding="utf-8-sig")
        if DEFAULT_CONFIG.target_column in data.columns:
            features = data.drop(columns=[DEFAULT_CONFIG.target_column])
        else:
            features = data

        if args.quantized_model:
            artifact = load_quantized_artifact(args.quantized_model, logger)
            transformed = model.named_steps["preprocess"].transform(features)
            selected = model.named_steps["select"].transform(transformed)
            labels, probabilities = predict_with_quantized_logits(selected, artifact)
        else:
            labels = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]

        result = features.copy()
        result["risk_label"] = pd.Series(labels).map({1: "Positive", 0: "Negative"})
        result["positive_probability"] = probabilities.round(4)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
        logger.info("Saved predictions to %s", args.output)
        print(json.dumps({"output": str(args.output), "rows": len(result)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.exception("Prediction command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

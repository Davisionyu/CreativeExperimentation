"""Lightweight int8 quantization for the logistic regression pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def quantize_logistic_pipeline(model: Pipeline, output_path: Path, logger: logging.Logger) -> dict[str, Any]:
    """Export int8 weights for a fitted logistic regression sklearn pipeline."""

    try:
        estimator = model.named_steps["model"]
        if not hasattr(estimator, "coef_"):
            raise TypeError("Quantization requires a linear model with coef_")

        weights = estimator.coef_[0].astype(np.float32)
        max_abs = float(np.max(np.abs(weights)))
        scale = max_abs / 127.0 if max_abs else 1.0
        weights_int8 = np.round(weights / scale).clip(-127, 127).astype(np.int8)
        artifact = {
            "format": "diabetes_logistic_int8_v1",
            "weight_scale": scale,
            "weights_int8": weights_int8.tolist(),
            "intercept": float(estimator.intercept_[0]),
            "classes": ["Negative", "Positive"],
            "note": "Use scripts/predict.py --quantized-model for inference; preprocessing is reused from the sklearn pipeline.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved quantized model artifact: %s", output_path)
        return artifact
    except Exception:
        logger.exception("Logistic pipeline quantization failed")
        raise


def predict_with_quantized_logits(
    transformed_features: np.ndarray,
    quantized_artifact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probabilities from already transformed selected features."""

    weights = np.asarray(quantized_artifact["weights_int8"], dtype=np.float32) * float(quantized_artifact["weight_scale"])
    logits = transformed_features @ weights + float(quantized_artifact["intercept"])
    probabilities = _sigmoid(logits)
    labels = (probabilities >= 0.5).astype(int)
    return labels, probabilities


def load_quantized_artifact(path: Path, logger: logging.Logger) -> dict[str, Any]:
    """Load an int8 model artifact from disk."""

    try:
        if not path.exists():
            raise FileNotFoundError(f"Quantized model artifact not found: {path}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("format") != "diabetes_logistic_int8_v1":
            raise ValueError(f"Unsupported quantized model format: {artifact.get('format')}")
        logger.info("Loaded quantized artifact: %s", path)
        return artifact
    except Exception:
        logger.exception("Loading quantized artifact failed: %s", path)
        raise

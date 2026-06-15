"""用于 Logistic Regression 管线的轻量 int8 量化。"""

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
    """导出已训练 Logistic Regression 管线的 int8 权重。"""

    try:
        estimator = model.named_steps["model"]
        if not hasattr(estimator, "coef_"):
            raise TypeError("量化需要带有 coef_ 的线性模型")

        weights = estimator.coef_[0].astype(np.float32)
        max_abs = float(np.max(np.abs(weights)))
        scale = max_abs / 127.0 if max_abs else 1.0
        weights_int8 = np.round(weights / scale).clip(-127, 127).astype(np.int8)
        artifact = {
            "format": "diabetes_logistic_int8_v1",
            "weight_scale": scale,
            "weights_int8": weights_int8.tolist(),
            "intercept": float(estimator.intercept_[0]),
            "classes": ["阴性", "阳性"],
            "note": "推理时使用 scripts/predict.py --quantized-model；预处理仍复用 sklearn 管线。",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("已保存量化模型产物：%s", output_path)
        return artifact
    except Exception:
        logger.exception("Logistic 管线量化失败")
        raise


def predict_with_quantized_logits(
    transformed_features: np.ndarray,
    quantized_artifact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """基于已变换后的特征预测标签与概率。"""

    weights = np.asarray(quantized_artifact["weights_int8"], dtype=np.float32) * float(quantized_artifact["weight_scale"])
    logits = transformed_features @ weights + float(quantized_artifact["intercept"])
    probabilities = _sigmoid(logits)
    labels = (probabilities >= 0.5).astype(int)
    return labels, probabilities


def load_quantized_artifact(path: Path, logger: logging.Logger) -> dict[str, Any]:
    """读取 int8 模型产物。"""

    try:
        if not path.exists():
            raise FileNotFoundError(f"未找到量化模型产物：{path}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("format") != "diabetes_logistic_int8_v1":
            raise ValueError(f"不支持的量化模型格式：{artifact.get('format')}")
        logger.info("已读取量化模型产物：%s", path)
        return artifact
    except Exception:
        logger.exception("读取量化模型产物失败：%s", path)
        raise

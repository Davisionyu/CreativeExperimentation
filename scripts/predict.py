"""使用训练好的糖尿病风险预测模型进行推理。"""

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
from diabetes_prediction.inference import risk_label_from_probability, risk_note_from_probability
from diabetes_prediction.logging_utils import setup_logging
from diabetes_prediction.modeling import load_model
from diabetes_prediction.quantization import load_quantized_artifact, predict_with_quantized_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测 CSV 中每条样本的糖尿病风险。")
    parser.add_argument("--input", type=Path, default=DEFAULT_CONFIG.data_path, help="输入 CSV，默认使用项目数据集。")
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG.report_dir / "predictions.csv", help="输出 CSV 路径。")
    parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_dir / "best_model.joblib", help="已训练的 sklearn 模型路径。")
    parser.add_argument("--quantized-model", type=Path, default=None, help="可选的 int8 量化模型路径。")
    parser.add_argument("--include-features", action="store_true", help="输出完整特征列，默认只输出精简预测结果。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(DEFAULT_CONFIG.log_dir, "predict")
    try:
        if not args.input.exists():
            raise FileNotFoundError(f"未找到输入文件：{args.input}")
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

        result = features.copy() if args.include_features else pd.DataFrame({"样本编号": range(1, len(features) + 1)})
        result["风险标签"] = [risk_label_from_probability(float(probability)) for probability in probabilities]
        result["阳性概率"] = probabilities.round(4)
        result["结果说明"] = [risk_note_from_probability(float(probability)) for probability in probabilities]
        if "class" in result.columns:
            result = result.rename(columns={"class": "原始标签"})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
        logger.info("已保存预测结果：%s", args.output)
        print(json.dumps({"输出文件": str(args.output), "行数": len(result)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.exception("推理命令失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

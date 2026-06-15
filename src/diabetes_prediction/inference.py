"""推理输入输出工具。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from diabetes_prediction.config import DEFAULT_CONFIG
from diabetes_prediction.data import EXPECTED_COLUMNS


FEATURE_COLUMNS = [column for column in EXPECTED_COLUMNS if column != DEFAULT_CONFIG.target_column]

FIELD_LABELS = {
    "Age": "年龄",
    "Gender": "性别",
    "Polyuria": "多尿",
    "Polydipsia": "烦渴",
    "sudden weight loss": "突然体重下降",
    "weakness": "乏力",
    "Polyphagia": "多食",
    "Genital thrush": "生殖器念珠菌感染",
    "visual blurring": "视物模糊",
    "Itching": "瘙痒",
    "Irritability": "易怒",
    "delayed healing": "伤口愈合延迟",
    "partial paresis": "局部麻痹",
    "muscle stiffness": "肌肉僵硬",
    "Alopecia": "脱发",
    "Obesity": "肥胖",
}

YES_NO_COLUMNS = [column for column in FEATURE_COLUMNS if column not in {"Age", "Gender"}]


def read_feature_table(path: Path) -> pd.DataFrame:
    """读取 CSV 或 Excel 表格。"""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    raise ValueError("仅支持 CSV、XLSX 或 XLSM 文件。")


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """校验输入表格，并按训练字段顺序返回特征。"""

    feature_df = df.drop(columns=[DEFAULT_CONFIG.target_column], errors="ignore").copy()
    missing_columns = [column for column in FEATURE_COLUMNS if column not in feature_df.columns]
    if missing_columns:
        missing_text = "、".join(missing_columns)
        raise ValueError(f"输入数据缺少字段：{missing_text}")
    return feature_df[FEATURE_COLUMNS]


def predict_dataframe(model, df: pd.DataFrame, include_features: bool = False) -> pd.DataFrame:
    """对表格数据进行预测并返回结果。"""

    features = normalize_input_columns(df)
    labels = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    result = features.copy() if include_features else pd.DataFrame({"样本编号": range(1, len(features) + 1)})
    result["风险标签"] = pd.Series(labels).map({1: "阳性", 0: "阴性"})
    result["阳性概率"] = probabilities.round(4)
    return result


def build_manual_record(form_data: dict[str, str]) -> pd.DataFrame:
    """把前端表单数据转换为模型输入的一行表格。"""

    record: dict[str, object] = {}
    for column in FEATURE_COLUMNS:
        value = form_data.get(column, "").strip()
        if column == "Age":
            if not value:
                raise ValueError("年龄不能为空。")
            record[column] = int(value)
        else:
            if not value:
                raise ValueError(f"{FIELD_LABELS.get(column, column)}不能为空。")
            record[column] = value
    return pd.DataFrame([record], columns=FEATURE_COLUMNS)

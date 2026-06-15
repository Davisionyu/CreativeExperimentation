"""推理输入输出工具。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from diabetes_prediction.config import DEFAULT_CONFIG
from diabetes_prediction.data import EXPECTED_COLUMNS


FEATURE_COLUMNS = [column for column in EXPECTED_COLUMNS if column != DEFAULT_CONFIG.target_column]

FIELD_LABELS = {
    "gender": "性别",
    "age": "年龄",
    "hypertension": "高血压",
    "heart_disease": "心脏病史",
    "smoking_history": "吸烟史",
    "bmi": "BMI",
    "HbA1c_level": "糖化血红蛋白",
    "blood_glucose_level": "血糖水平",
}

FIELD_OPTIONS = {
    "gender": [("Female", "女"), ("Male", "男"), ("Other", "其他")],
    "hypertension": [("0", "否"), ("1", "是")],
    "heart_disease": [("0", "否"), ("1", "是")],
    "smoking_history": [
        ("No Info", "未知"),
        ("never", "从不吸烟"),
        ("former", "曾经吸烟"),
        ("current", "当前吸烟"),
        ("not current", "目前不吸烟"),
        ("ever", "曾有吸烟史"),
    ],
}

NUMERIC_FIELDS = {
    "age": {"default": "45", "min": "0", "max": "120", "step": "1"},
    "bmi": {"default": "24.0", "min": "10", "max": "80", "step": "0.1"},
    "HbA1c_level": {"default": "5.7", "min": "3", "max": "12", "step": "0.1"},
    "blood_glucose_level": {"default": "120", "min": "50", "max": "400", "step": "1"},
}

DEFAULT_FORM_VALUES = {
    "gender": "Female",
    "age": "45",
    "hypertension": "0",
    "heart_disease": "0",
    "smoking_history": "No Info",
    "bmi": "24.0",
    "HbA1c_level": "5.7",
    "blood_glucose_level": "120",
}


def risk_label_from_probability(probability: float) -> str:
    """把阳性概率转换为更适合筛查场景的风险分层。"""

    if probability >= 0.7:
        return "风险偏高"
    if probability >= 0.5:
        return "需关注"
    return "风险较低"


def risk_note_from_probability(probability: float) -> str:
    """生成面向用户的风险解释。"""

    if probability >= 0.7:
        return "模型提示风险偏高，建议结合血糖检测和医生意见进一步确认。"
    if probability >= 0.5:
        return "模型结果接近判断边界，表示需要关注，不等同于已经确诊。"
    return "模型提示当前风险较低，但仍不能替代医学检查。"


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
    probabilities = model.predict_proba(features)[:, 1]

    result = features.copy() if include_features else pd.DataFrame({"样本编号": range(1, len(features) + 1)})
    result["风险标签"] = [risk_label_from_probability(float(probability)) for probability in probabilities]
    result["阳性概率"] = probabilities.round(4)
    result["结果说明"] = [risk_note_from_probability(float(probability)) for probability in probabilities]
    return result


def build_manual_record(form_data: dict[str, str]) -> pd.DataFrame:
    """把前端表单数据转换为模型输入的一行表格。"""

    record: dict[str, object] = {}
    for column in FEATURE_COLUMNS:
        value = form_data.get(column, "").strip()
        if column in NUMERIC_FIELDS:
            if not value:
                raise ValueError(f"{FIELD_LABELS.get(column, column)}不能为空。")
            record[column] = float(value)
        elif column in {"hypertension", "heart_disease"}:
            if value not in {"0", "1"}:
                raise ValueError(f"{FIELD_LABELS.get(column, column)}只能选择是或否。")
            record[column] = int(value)
        else:
            if not value:
                raise ValueError(f"{FIELD_LABELS.get(column, column)}不能为空。")
            record[column] = value
    return pd.DataFrame([record], columns=FEATURE_COLUMNS)

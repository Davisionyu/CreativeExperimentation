"""糖尿病风险预测 Web 应用。"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, send_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diabetes_prediction.config import DEFAULT_CONFIG
from diabetes_prediction.inference import (
    DEFAULT_FORM_VALUES,
    FIELD_LABELS,
    FIELD_OPTIONS,
    FEATURE_COLUMNS,
    NUMERIC_FIELDS,
    build_manual_record,
    predict_dataframe,
    read_feature_table,
)
from diabetes_prediction.logging_utils import setup_logging
from diabetes_prediction.modeling import load_model

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
logger = setup_logging(DEFAULT_CONFIG.log_dir, "web_app")


def _load_best_model():
    return load_model(DEFAULT_CONFIG.model_dir / "best_model.joblib", logger)


def _table_payload(df: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": df.columns.tolist(),
        "rows": df.to_dict(orient="records"),
        "total_rows": len(df),
    }


def _default_manual_form() -> dict[str, str]:
    return DEFAULT_FORM_VALUES.copy()


def _manual_form_from_request() -> dict[str, str]:
    form_values = _default_manual_form()
    for column in FEATURE_COLUMNS:
        if column in request.form:
            form_values[column] = request.form.get(column, "")
    return form_values


@app.get("/")
def index():
    return render_template(
        "index.html",
        feature_columns=FEATURE_COLUMNS,
        field_labels=FIELD_LABELS,
        field_options=FIELD_OPTIONS,
        numeric_fields=NUMERIC_FIELDS,
        manual_form=_default_manual_form(),
    )


@app.post("/predict/manual")
def predict_manual():
    try:
        model = _load_best_model()
        input_df = build_manual_record(request.form)
        result = predict_dataframe(model, input_df)
        manual_form = _manual_form_from_request()
        return render_template(
            "index.html",
            feature_columns=FEATURE_COLUMNS,
            field_labels=FIELD_LABELS,
            field_options=FIELD_OPTIONS,
            numeric_fields=NUMERIC_FIELDS,
            manual_form=manual_form,
            manual_result=result.iloc[0].to_dict(),
            active_panel="manual",
        )
    except Exception as exc:
        logger.exception("手动预测失败：%s", exc)
        return render_template(
            "index.html",
            feature_columns=FEATURE_COLUMNS,
            field_labels=FIELD_LABELS,
            field_options=FIELD_OPTIONS,
            numeric_fields=NUMERIC_FIELDS,
            manual_form=_manual_form_from_request(),
            error=str(exc),
            active_panel="manual",
        ), 400


@app.post("/predict/file")
def predict_file():
    try:
        upload = request.files.get("data_file")
        if upload is None or not upload.filename:
            raise ValueError("请先选择 CSV 或 Excel 文件。")

        suffix = Path(upload.filename).suffix.lower()
        temp_path = DEFAULT_CONFIG.report_dir / f"upload_input{suffix}"
        DEFAULT_CONFIG.report_dir.mkdir(parents=True, exist_ok=True)
        upload.save(temp_path)

        model = _load_best_model()
        input_df = read_feature_table(temp_path)
        result = predict_dataframe(model, input_df)
        output_path = DEFAULT_CONFIG.report_dir / "web_predictions.csv"
        result.to_csv(output_path, index=False, encoding="utf-8-sig")

        return render_template(
            "index.html",
            feature_columns=FEATURE_COLUMNS,
            field_labels=FIELD_LABELS,
            field_options=FIELD_OPTIONS,
            numeric_fields=NUMERIC_FIELDS,
            manual_form=_default_manual_form(),
            file_result=_table_payload(result),
            download_ready=True,
            active_panel="file",
        )
    except Exception as exc:
        logger.exception("文件预测失败：%s", exc)
        return render_template(
            "index.html",
            feature_columns=FEATURE_COLUMNS,
            field_labels=FIELD_LABELS,
            field_options=FIELD_OPTIONS,
            numeric_fields=NUMERIC_FIELDS,
            manual_form=_default_manual_form(),
            error=str(exc),
            active_panel="file",
        ), 400


@app.get("/download")
def download_predictions():
    output_path = DEFAULT_CONFIG.report_dir / "web_predictions.csv"
    if output_path.exists():
        return send_file(output_path, as_attachment=True, download_name="糖尿病风险预测结果.csv")
    empty = BytesIO("暂无预测结果。".encode("utf-8"))
    return send_file(empty, as_attachment=True, download_name="糖尿病风险预测结果.csv", mimetype="text/csv")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

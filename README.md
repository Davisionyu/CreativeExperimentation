# 糖尿病预测项目

本项目实现一个可复现的糖尿病早期风险预测系统，包含数据预处理、特征选择、模型训练、推理、int8 量化和实验报告材料。核心入口均包含异常处理和日志记录，日志输出到 `logs/`。

## 数据与来源

本地数据文件为 `diabetes_prediction_dataset.csv`，包含 100000 条样本、8 个输入特征和 1 个二分类标签 `diabetes`。字段包括 `age`、`gender`、`bmi`、`hypertension`、`heart_disease`、`smoking_history`、`HbA1c_level`、`blood_glucose_level` 等。数据来源为 Kaggle Diabetes prediction dataset：https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

说明：原始数据字段保留英文，这是为了与公开数据集字段保持一致，避免训练、推理和引用来源不一致。旧的 `糖尿病预测.csv` 保留在仓库中作为实训原始数据备份。

## 项目结构

```text
.
├── scripts/
│   ├── train.py          # 训练、调参、评估、误判样本输出
│   ├── predict.py        # 普通模型和量化模型推理
│   └── quantize.py       # 训练 Logistic 模型并导出 int8 权重
├── src/diabetes_prediction/
│   ├── data.py           # 数据读取、校验、划分
│   ├── features.py       # 统一维度预处理和特征选择
│   ├── modeling.py       # 模型训练、评估、持久化
│   └── quantization.py   # int8 量化和量化推理
├── reports/
│   ├── experiment_report.md
│   └── architecture.mmd
├── tests/
├── diabetes_prediction_dataset.csv
└── 糖尿病预测.csv
```

## 算法流程

1. 读取 CSV 并自动尝试常见编码。
2. 校验字段完整性、目标列、缺失值和重复样本。
3. 将不同量纲数据统一处理：年龄、BMI、HbA1c、血糖等数值特征使用中位数填充和标准化，性别、吸烟史等类别特征使用众数填充和独热编码。
4. 使用互信息 `mutual_info_classif` 选择前 K 个特征，减少冗余和噪声。
5. 训练并比较逻辑回归、随机森林、梯度提升三类模型。
6. 使用 5 折分层交叉验证，以 F1 作为主指标选择最优模型。
7. 输出测试集指标、特征重要性、误判样本、最优权重和量化权重。

## 参数设置

默认参数集中在 `src/diabetes_prediction/config.py`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `test_size` | `0.2` | 测试集比例 |
| `validation_size` | `0.2` | 验证集比例 |
| `cv_folds` | `5` | 交叉验证折数 |
| `feature_k` | `10` | 互信息选择的特征数量 |
| `random_state` | `42` | 随机种子 |

## 复现步骤

```bash
pip install -r requirements.txt
pip install -e .
python scripts/train.py
python scripts/quantize.py
python scripts/predict.py --input diabetes_prediction_dataset.csv --output reports/predictions.csv
python scripts/predict.py --input diabetes_prediction_dataset.csv --model models/logistic_model.joblib --quantized-model models/logistic_int8.json --output reports/predictions_int8.csv
python scripts/web_app.py
pytest
```

训练完成后会生成：

- `models/best_model.joblib`：交叉验证 F1 最优模型。
- `models/logistic_model.joblib`：用于量化推理的 Logistic 管线。
- `models/logistic_int8.json`：int8 量化权重。
- `reports/metrics.json`：验证集、测试集和交叉验证指标。
- `reports/feature_importance.csv`：特征选择分数与模型重要性。
- `reports/badcases.csv`：测试集误判样本。

## 推理示例

```bash
python scripts/predict.py --input diabetes_prediction_dataset.csv --output reports/predictions.csv
```

启动前后端页面：

```bash
python scripts/web_app.py
```

浏览器访问 `http://127.0.0.1:5000`。页面支持手动填写年龄、BMI、HbA1c、血糖、高血压、心脏病史、吸烟史等指标，也支持上传 CSV、XLSX 或 XLSM 表格进行批量预测，批量结果可下载为 CSV。

量化推理：

```bash
python scripts/predict.py --input diabetes_prediction_dataset.csv --model models/logistic_model.joblib --quantized-model models/logistic_int8.json --output reports/predictions_int8.csv
```

## 学术诚信说明

本项目代码为独立实现，未复制第三方项目代码。数据集引用 Kaggle Diabetes prediction dataset；模型算法基于 scikit-learn 官方库，包括逻辑回归、随机森林、梯度提升、独热编码、标准化和互信息特征选择。实验报告中涉及数据集、论文算法或开源实现时应保留明确引用。

# 糖尿病预测实验报告

## 1. 项目概述

本项目面向糖尿病早期风险识别任务，使用结构化健康指标数据训练二分类模型，输出患者是否存在糖尿病风险及阳性概率。系统覆盖数据校验、特征处理、模型训练、模型评估、推理部署、量化推理和 Badcase 分析。

## 2. 数据来源与学术诚信

本地数据文件为 `糖尿病预测.csv`，字段与 UCI Machine Learning Repository 的 Early Stage Diabetes Risk Prediction Dataset 一致。该公开数据集包含 520 条样本、16 个特征和 1 个二分类标签，数据由孟加拉国 Sylhet Diabetes Hospital 患者问卷采集并经医生确认。引用来源：https://archive.ics.uci.edu/dataset/529/early%2Bstage%2Bdiabetes%2Brisk%2Bprediction%2Bdataset

本项目代码由本仓库独立实现，未复制第三方项目代码。算法使用 scikit-learn 官方实现，实验报告和 README 中均标注了数据和算法来源。

## 3. 系统架构

```mermaid
flowchart LR
    A["Raw CSV"] --> B["Data Validation"]
    B --> C["Preprocessing"]
    C --> D["Feature Selection"]
    D --> E["Model Tuning"]
    E --> F["Best Model"]
    F --> G["Inference"]
    F --> H["Metrics + Badcases"]
    C --> I["Logistic Model"]
    I --> J["Int8 Quantized Model"]
```

## 4. 算法流程

1. 数据读取：自动尝试 `utf-8-sig`、`utf-8`、`gb18030`、`gbk` 编码。
2. 数据校验：检查必要字段、空数据、目标列二分类属性、缺失值和重复样本。
3. 预处理：年龄字段进行中位数填充和标准化，类别字段进行众数填充和 One-Hot 编码。
4. 特征处理：使用互信息 `mutual_info_classif` 选择 Top-K 重要特征，降低冗余和噪声。
5. 模型训练：比较 Logistic Regression、Random Forest、Gradient Boosting，并通过 Stratified K-Fold 交叉验证选择 F1 最优模型。
6. 评估输出：生成 Accuracy、Precision、Recall、F1、ROC-AUC、混淆矩阵、特征重要性和 Badcase 文件。
7. 量化部署：训练 Logistic Regression 推理模型，将权重量化为 int8 JSON，保留 sklearn 预处理管线用于一致推理。

## 5. 参数设置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `test_size` | `0.2` | 测试集比例 |
| `validation_size` | `0.2` | 验证集比例 |
| `cv_folds` | `5` | 交叉验证折数 |
| `feature_k` | `12` | 互信息选择的特征数量 |
| `random_state` | `42` | 随机种子 |

## 6. 复现步骤

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/quantize.py
python scripts/predict.py --input 糖尿病预测.csv --output reports/predictions.csv
python scripts/predict.py --input 糖尿病预测.csv --model models/logistic_model.joblib --quantized-model models/logistic_int8.json --output reports/predictions_int8.csv
pytest
```

## 7. 实验结果填写说明

运行 `python scripts/train.py` 后，主要结果位于：

- `reports/metrics.json`：交叉验证、验证集、测试集指标。
- `reports/feature_importance.csv`：被选择特征、互信息分数和模型重要性。
- `reports/badcases.csv`：测试集错误样本及预测概率。
- `models/best_model.joblib`：最优模型权重。

## 8. Badcase 分析方法

优先检查 `reports/badcases.csv` 中阳性概率接近 0.5 的样本，这类样本通常处于决策边界附近。再结合 `feature_importance.csv` 分析 Polyuria、Polydipsia、Gender、Age 等关键变量是否出现互相矛盾的症状组合。由于本数据集规模较小，Badcase 结论应作为模型改进线索，而不能直接替代医学诊断。

## 9. 后续优化

可进一步引入外部验证集、医学成本敏感阈值调优、SHAP 可解释性分析、模型校准和 Web/API 部署。医疗风险场景中，Recall 通常应被优先关注，以降低漏检风险。

"""项目统一配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    """默认路径和训练参数。"""

    data_path: Path = PROJECT_ROOT / "diabetes_prediction_dataset.csv"
    target_column: str = "diabetes"
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 5
    feature_k: int = 10
    positive_label: int = 1
    model_dir: Path = PROJECT_ROOT / "models"
    report_dir: Path = PROJECT_ROOT / "reports"
    log_dir: Path = PROJECT_ROOT / "logs"


DEFAULT_CONFIG = ProjectConfig()

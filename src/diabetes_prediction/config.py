"""项目统一配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """默认路径和训练参数。"""

    data_path: Path = Path("糖尿病预测.csv")
    target_column: str = "class"
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 5
    feature_k: int = 12
    positive_label: str = "Positive"
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    log_dir: Path = Path("logs")


DEFAULT_CONFIG = ProjectConfig()

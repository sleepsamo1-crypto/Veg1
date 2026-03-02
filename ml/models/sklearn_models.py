"""Sklearn models for vegetable price prediction"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from .base import BaseModel


class SklearnModel(BaseModel):
    def __init__(self, name: str, model):
        # 调用父类，传入空字典作为 params（如果不需要在父类保存额外参数）
        super().__init__(name, {})
        self.model = model   # 关键：将模型对象赋给 self.model
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_object(self):
        return self.model


def build_sklearn_models(config: dict) -> dict:
    """构建所有sklearn模型
    
    Args:
        config: 超参数配置字典
    
    Returns:
        {model_name: model_instance}
    """
    models = {}
    
    # Ridge回归
    models["Ridge"] = SklearnModel(
        "Ridge",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(**config["Ridge"]))
        ])
    )
    
    # 随机森林
    models["RandomForest"] = SklearnModel(
        "RandomForest",
        RandomForestRegressor(**config["RandomForest"])
    )
    
    # ExtraTrees
    models["ExtraTrees"] = SklearnModel(
        "ExtraTrees",
        ExtraTreesRegressor(**config["ExtraTrees"])
    )
    
    # GradientBoosting
    models["GradientBoosting"] = SklearnModel(
        "GradientBoosting",
        GradientBoostingRegressor(**config["GradientBoosting"])
    )
    
    # HistGradientBoosting
    models["HistGradientBoosting"] = SklearnModel(
        "HistGradientBoosting",
        HistGradientBoostingRegressor(**config["HistGradientBoosting"])
    )
    
    return models
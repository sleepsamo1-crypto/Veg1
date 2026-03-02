"""XGBoost model for vegetable price prediction"""
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost模型包装器"""
    
    def __init__(self, params: dict):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
        super().__init__("XGBoost", params)
        self.model = XGBRegressor(**params)
    
    def fit(self, X_train, y_train):
        """训练XGBoost模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        self.model.fit(
            X_train, y_train,
            verbose=False,
        )
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model_object(self):
        return self.model


def build_xgboost_model(config: dict) -> XGBoostModel:
    """构建XGBoost模型"""
    return XGBoostModel(config["XGBoost"])
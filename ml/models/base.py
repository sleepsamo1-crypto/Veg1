"""Base class for all models"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseModel(ABC):
    """所有模型的基类"""
    
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.model = None
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    @abstractmethod
    def get_model_object(self):
        """返回原始模型对象（用于保存）"""
        pass
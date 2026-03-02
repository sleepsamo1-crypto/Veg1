"""LSTM model for vegetable price prediction"""
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

import numpy as np
from .base import BaseModel


class LSTMModel(BaseModel):
    """LSTM模型包装器"""
    
    def __init__(self, params: dict):
        if not LSTM_AVAILABLE:
            raise ImportError(
                "TensorFlow not installed. Install with: pip install tensorflow"
            )
        super().__init__("LSTM", params)
        self.model = self._build_model(params)
    
    def _build_model(self, params: dict):
        """构建LSTM模型"""
        model = keras.Sequential([
            layers.LSTM(
                params["lstm_units_1"],
                activation="relu",
                input_shape=(1, params["input_shape"]),
                return_sequences=True,
            ),
            layers.Dropout(params["dropout_1"]),
            layers.LSTM(params["lstm_units_2"], activation="relu"),
            layers.Dropout(params["dropout_2"]),
            layers.Dense(params["dense_units"], activation="relu"),
            layers.Dropout(params["dropout_3"]),
            layers.Dense(1)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=params["learning_rate"])
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        return model
    
    def fit(self, X_train, y_train):
        """训练LSTM"""
        # LSTM需要3D输入 (samples, timesteps, features)
        X_train_3d = np.expand_dims(X_train, axis=1)
        
        self.model.fit(
            X_train_3d, y_train,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=0.2,
            verbose=0,
        )
    
    def predict(self, X):
        # 转换为3D输入
        X_3d = np.expand_dims(X, axis=1)
        return self.model.predict(X_3d, verbose=0).flatten()
    
    def get_model_object(self):
        return self.model


def build_lstm_model(params: dict) -> LSTMModel:
    """构建LSTM模型"""
    return LSTMModel(params["LSTM"])
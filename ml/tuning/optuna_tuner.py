"""Automatic hyperparameter tuning using Optuna

Optional: 自动调参工具，可用于找到最优参数
"""

try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")


class HyperparameterTuner:
    """使用Optuna进行自动调参"""
    
    def __init__(self, model_name: str, X_train, y_train, n_trials: int = 20):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna not installed. Install with: pip install optuna"
            )
        
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.n_trials = n_trials
    
    def objective_xgboost(self, trial):
        """XGBoost的目标函数"""
        from xgboost import XGBRegressor
        
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42,
        }
        
        model = XGBRegressor(**params, verbosity=0)
        
        # 5折交叉验证
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring="neg_mean_squared_error"
        )
        
        return -scores.mean()  # 返回负数以供最小化
    
    def objective_random_forest(self, trial):
        """RandomForest的目标函数"""
        from sklearn.ensemble import RandomForestRegressor
        
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 10, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42,
            "n_jobs": -1,
        }
        
        model = RandomForestRegressor(**params)
        
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring="neg_mean_squared_error"
        )
        
        return -scores.mean()
    
    def tune(self) -> dict:
        """执行调参
        
        Returns:
            最优参数字典
        """
        
        if self.model_name == "XGBoost":
            objective = self.objective_xgboost
        elif self.model_name == "RandomForest":
            objective = self.objective_random_forest
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = MedianPruner()
        
        study = optuna.create_study(sampler=sampler, pruner=pruner)
        
        print(f"[Optuna] 开始调参 {self.model_name}...")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"\n✓ 最优参数: {study.best_params}")
        print(f"✓ 最佳得分: {-study.best_value:.6f}")
        
        return study.best_params


# 使用示例
if __name__ == "__main__":
    print("Optuna hyperparameter tuning example")
    print("$ pip install optuna")
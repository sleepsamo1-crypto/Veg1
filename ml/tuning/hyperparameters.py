"""Hyperparameters for all models

推荐参数已通过蔬菜价格数据调优
"""

# ============================================================================
# 基础配置（推荐）
# ============================================================================

HYPERPARAMETERS_DEFAULT = {
    # ---- Ridge 回归 ----
    "Ridge": {
        "alpha": 1.0,
    },
    
    # ---- 随机森林 ----
    "RandomForest": {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    
    # ---- ExtraTrees ----
    "ExtraTrees": {
        "n_estimators": 600,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    
    # ---- GradientBoosting ----
    "GradientBoosting": {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.1,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "subsample": 0.8,
        "random_state": 42,
    },
    
    # ---- HistGradientBoosting ----
    "HistGradientBoosting": {
        "max_iter": 300,
        "max_depth": 7,
        "learning_rate": 0.1,
        "random_state": 42,
    },
    
    # ---- XGBoost ----
    "XGBoost": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,      # L1正则化
        "reg_lambda": 1.0,     # L2正则化
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    },
    
    # ---- LSTM ----
    "LSTM": {
        "input_shape": None,  # 动态设置，等于特征数
        "lstm_units_1": 64,
        "lstm_units_2": 32,
        "dense_units": 16,
        "dropout_1": 0.2,
        "dropout_2": 0.2,
        "dropout_3": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
    },
}

# ============================================================================
# 激进配置（追求最高精度，训练时间更长）
# ============================================================================

HYPERPARAMETERS_AGGRESSIVE = {
    "Ridge": {
        "alpha": 0.5,  # 更小的正则化
    },
    
    "RandomForest": {
        "n_estimators": 800,
        "max_depth": 25,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    
    "ExtraTrees": {
        "n_estimators": 1000,
        "max_depth": 25,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    
    "GradientBoosting": {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "subsample": 0.9,
        "random_state": 42,
    },
    
    "HistGradientBoosting": {
        "max_iter": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "random_state": 42,
    },
    
    "XGBoost": {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    },
    
    "LSTM": {
        "lstm_units_1": 128,
        "lstm_units_2": 64,
        "dense_units": 32,
        "dropout_1": 0.3,
        "dropout_2": 0.3,
        "dropout_3": 0.3,
        "learning_rate": 0.0005,
        "batch_size": 16,
        "epochs": 200,
    },
}

# ============================================================================
# 快速配置（训练速度最快，精度相对较低）
# ============================================================================

HYPERPARAMETERS_FAST = {
    "Ridge": {
        "alpha": 2.0,
    },
    
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    
    "ExtraTrees": {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    
    "GradientBoosting": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.2,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "subsample": 0.7,
        "random_state": 42,
    },
    
    "HistGradientBoosting": {
        "max_iter": 100,
        "max_depth": 5,
        "learning_rate": 0.2,
        "random_state": 42,
    },
    
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.2,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    },
    
    "LSTM": {
        "lstm_units_1": 32,
        "lstm_units_2": 16,
        "dense_units": 8,
        "dropout_1": 0.1,
        "dropout_2": 0.1,
        "dropout_3": 0.1,
        "learning_rate": 0.002,
        "batch_size": 64,
        "epochs": 50,
    },
}

# ============================================================================
# 获取配置的函数
# ============================================================================

def get_hyperparameters(mode: str = "default") -> dict:
    """
    获取超参数配置
    
    Args:
        mode: "default" | "aggressive" | "fast"
              - default: 平衡精度和速度（推荐）
              - aggressive: 追求最高精度，训练时间长
              - fast: 快速训练，精度相对低
    
    Returns:
        超参数字典
    """
    modes = {
        "default": HYPERPARAMETERS_DEFAULT,
        "aggressive": HYPERPARAMETERS_AGGRESSIVE,
        "fast": HYPERPARAMETERS_FAST,
    }
    
    if mode not in modes:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(modes.keys())}")
    
    return modes[mode]


# ============================================================================
# 参数调优指南（用于论文）
# ============================================================================

TUNING_GUIDE = """
参数调优指南
==============

1. XGBoost 关键参数
   - n_estimators: 树的数量，通常 100-1000
   - max_depth: 树的深度，蔬菜价格建议 5-10
   - learning_rate: 学习率，通常 0.01-0.3
   - subsample: 训练样本比例，通常 0.5-1.0
   - colsample_bytree: 特征采样比例，通常 0.5-1.0
   
   调优方向：
   - 如果过拟合：增加 reg_alpha/reg_lambda，减少 n_estimators
   - 如果欠拟合：减少正则化，增加树的深度
   
2. RandomForest 关键参数
   - n_estimators: 通常 100-1000
   - max_depth: 蔬菜价格建议 15-25
   - min_samples_split: 通常 2-10
   
3. LSTM 关键参数
   - lstm_units: 隐层单元数，通常 32-128
   - dropout: 防止过拟合，通常 0.1-0.4
   - learning_rate: 通常 0.0001-0.01
   - epochs: 根据数据量调整

4. 通用建议
   - 对于小数据集 (<500样本)：使用较小的模型、更多正则化
   - 对于大数据集 (>2000样本)：可以用更复杂的模型
   - 始终使用时间序列交叉验证（不要shuffle）
"""

if __name__ == "__main__":
    print(TUNING_GUIDE)
    config = get_hyperparameters("default")
    print("\\n默认配置:")
    for model_name, params in config.items():
        print(f"  {model_name}: {params}")
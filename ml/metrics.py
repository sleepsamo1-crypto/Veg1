import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def mase(y_true, y_pred, training_data):
    naive_forecast = training_data[:-1]
    mae_naive = np.mean(np.abs(training_data[1:] - naive_forecast))
    return mae(y_true, y_pred) / mae_naive

# Example usage (uncomment to use):
# y_true = np.array([1, 2, 3])
# y_pred = np.array([1, 2, 4])
# print(f"RMSE: {rmse(y_true, y_pred)}")
# print(f"MAE: {mae(y_true, y_pred)}")
# print(f"MAPE: {mape(y_true, y_pred)}")
# print(f"SMAPE: {smape(y_true, y_pred)}")
# print(f"MASE: {mase(y_true, y_pred, np.array([1, 2, 1]))}")
import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100


def mase(y_true, y_pred, y_train):
    n = len(y_true)
    d = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / d

# Example usage
if __name__ == '__main__':
    # Actual and predicted values
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    print('RMSE:', rmse(y_true, y_pred))
    print('MAE:', mae(y_true, y_pred))
    print('MAPE:', mape(y_true, y_pred))
    print('SMAPE:', smape(y_true, y_pred))
    print('MASE:', mase(y_true, y_pred, np.array([3, 0, 2, 7])))
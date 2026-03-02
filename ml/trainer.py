from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


def train_model(df):
    """训练一个简单的回归模型"""
    X = df[["month", "day"]]  # 选择相关特征
    y = df["avg_price"]  # 目标变量

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 使用线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(model, "model.pkl")
    return model

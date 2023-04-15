from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 定义自变量X和因变量y
data = pd.read_csv('mmc2.csv', index_col=0) #列索引为第一行
# print(data.shape)
X = data.iloc[1:, :-1]
y = data.iloc[1:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建Bootstrap函数
def bootstrap(X, y):
    n_iterations = 100
    n_size = int(len(X) * 0.7)
    stats = []
    for i in range(n_iterations):
        X_resampled, y_resampled = resample(X, y, n_samples=n_size)
        stats.append((X_resampled, y_resampled))
    return stats

# 定义多项式回归模型


# 创建Bootstrap数据集
bootstrapped_data = bootstrap(X, y)
# print(bootstrapped_data)
rmses = []
for i in range(100):
    poly = PolynomialFeatures(degree=2)
    lm = LinearRegression()
    X_train, y_train = bootstrapped_data[i]
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    lm.fit(X_train_poly, y_train)
    y_pred = lm.predict(X_test_poly)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)

# 计算平均RMSE分数
score = np.mean(rmses)
print('RMSE of bootstrap:', score)

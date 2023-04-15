import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd



# 读取数据集
data = pd.read_csv('mmc2.csv', index_col=0) #列索引为第一行
# print(data.shape)
X = data.iloc[1:, :-1]
y = data.iloc[1:, -1]

# 变化训练集和测试集划分比例对模型性能的影响
ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
errors = []
for ratio in ratios:
    ratio_errors = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)

    # 准备用于网格搜索的参数
    params = [{'C': [0.01, 0.1, 0.03, 1, 10, 50, 100 ], 'epsilon': [0.001, 0.003,  0.01, 0.03, 0.1, 0.3, 0.5]}]
    # 创建SVR对象
    model = SVR(kernel='linear')

    # 创建GridSearch对象
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')

    # 训练网格搜索对象
    grid_search.fit(X_train, y_train)
    # 使用最佳模型进行预测
    best_svr = grid_search.best_estimator_
    y_pred = best_svr.predict(X_test)

    # 计算预测误
    test_error = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(test_error)
    ratio_errors.append(rmse)
    # 输出最佳参数和最佳模型
    print("Best parameters: ", grid_search.best_params_)
    # print("Best estimator: ", grid_search.best_estimator_)
    print(ratio_errors)

# 将不同的划分比例和预测误差结果打印出来
for ratio, error in zip(ratios, errors):
    print(f" {model}模型{ratio*100}%下训练集的RMSE: {error}")

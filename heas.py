import pandas as pd
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

data = pd.read_csv('mmc2.csv', index_col=0) #列索引为第一行
# print(data.shape)
X = data.iloc[1:, :-1]
y = data.iloc[1:, -1]
# 变化训练集和测试集划分比例对模型性能的影响
ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for ratio in ratios:
    errors = {
        'lin': [],
        'poly': [],
        'svr.l': [],
        'svr.p': [],
        'svr.r': [],
        'cart': [],
        'bpnn': [],
        'knn': []
    }
    ratio_errors = []
    #100次平均rmse
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)

        #lr
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        ratio_errors.append(rmse)
    errors['lin'].append(np.mean(ratio_errors))
    # print(errors)


    # for i in range(100):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)
    #
    #     #poly
    #     poly = PolynomialFeatures(degree=2)
    #     X_train_poly = poly.fit_transform(X_train)
    #     X_test_poly = poly.transform(X_test)
    #     model = LinearRegression
    #     model.fit(X_train_poly, y_train)
    #     y_pred = model.predict(X_test_poly)
    #     mse = mean_squared_error(y_test, y_pred)
    #     rmse = np.sqrt(mse)
    #     ratio_errors.append(rmse)
    # errors['poly'].append(np.mean(ratio_errors))

    # svrl
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)
        model = SVR(kernel='linear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        ratio_errors.append(rmse)
    errors['svr.l'].append(np.mean(ratio_errors))

    # svrr
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)

        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        ratio_errors.append(rmse)
    errors['svr.r'].append(np.mean(ratio_errors))

    # svrp
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)

        model = SVR(kernel='poly')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        ratio_errors.append(rmse)
    errors['svr.p'].append(np.mean(ratio_errors))

    # tree
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        ratio_errors.append(rmse)
    errors['cart'].append(np.mean(ratio_errors))

    # # bpnn
    # for i in range(100):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)
    #
    #     model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100000)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     mse = mean_squared_error(y_test, y_pred)
    #     rmse = np.sqrt(mse)
    #     ratio_errors.append(rmse)
    # errors['bpnn'].append(np.mean(ratio_errors))

    # knn
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)

        model = KNeighborsRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        ratio_errors.append(rmse)
    errors['knn'].append(np.mean(ratio_errors))
    print(errors)

# plt.plot(ratios, errors['poly'], 'o-', color='blue', label= "poly")
# plt.plot(ratios, errors['lr'], 'o-',  color='red', label= "lr")
# plt.plot(ratios, errors['bpnn'], 'o-',  color='black', label= "bpnn")
# plt.plot(train_size, rmses_cart, 'o-',  color='green', label= "cart")
# plt.plot(train_size, rmses_knn, 'o-',  color='y', label= "knn")
# plt.plot(train_size, rmses_svrl, 'o-',  color='m', label= "svrl")
# plt.plot(train_size, rmses_svrp, 'o-',  color='c', label= "svrp")
# plt.plot(train_size, rmses_svrr, 'o-',  color='aqua', label= "svrr")
# plt.title('LinearRegression')
# plt.xlabel('train_size')
# plt.ylabel('RMSE')
# plt.legend()
# plt.show()



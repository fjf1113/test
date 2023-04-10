import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv('mmc2.csv', index_col=0) #列索引为第一行
# print(data.shape)
X = data.iloc[1:, :-1]
y = data.iloc[1:, -1]
test_size_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
train_size = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3] #训练集样本的占比

#线性回归rmse模型
rmses_lr = []
for test_size in test_size_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses_lr.append(rmse)
print(rmses_lr)

#svrr模型
rmses_svrr = []
for test_size in test_size_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses_svrr.append(rmse)
print(rmses_svrr)

#svrp模型
rmses_svrp = []
for test_size in test_size_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model = SVR(kernel='poly')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses_svrp.append(rmse)
print(rmses_svrp)

#svrl模型
rmses_svrl = []
for test_size in test_size_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses_svrl.append(rmse)
print(rmses_svrl)

#knn模型
rmses_knn = []
for test_size in test_size_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses_knn.append(rmse)
print(rmses_knn)

#cart决策树模型
rmses_cart = []
for test_size in test_size_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses_cart.append(rmse)
print(rmses_cart)

#bpnn模型
rmses_bpnn = []
for test_size in test_size_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model =MLPClassifier(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses_bpnn.append(rmse)
print(rmses_bpnn)

#poly模型
rmses_poly = []
for test_size in test_size_list:
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=0)

    model = make_pipeline(LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmses_poly.append(rmse)
print(rmses_poly)


plt.plot(train_size, rmses_poly, 'o-', color='blue', label= "poly")
plt.plot(train_size, rmses_lr, 'o-',  color='red', label= "lr")
plt.plot(train_size, rmses_bpnn, 'o-',  color='black', label= "bpnn")
plt.plot(train_size, rmses_cart, 'o-',  color='green', label= "cart")
plt.plot(train_size, rmses_knn, 'o-',  color='y', label= "knn")
plt.plot(train_size, rmses_svrl, 'o-',  color='m', label= "svrl")
plt.plot(train_size, rmses_svrp, 'o-',  color='c', label= "svrp")
plt.plot(train_size, rmses_svrr, 'o-',  color='aqua', label= "svrr")
plt.title('LinearRegression')
plt.xlabel('train_size')
plt.ylabel('RMSE')
plt.legend()
plt.show()



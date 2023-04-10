from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#poly模型
data = pd.read_csv('mmc2.csv', index_col=0) #列索引为第一行
# print(data.shape)
X = data.iloc[1:, :-1]
y = data.iloc[1:, -1]

rmses_poly = []
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

model = make_pipeline(LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmses_poly.append(rmse)
print(rmses_poly)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.show()
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("phone_sale.csv")
data = data.dropna()
x5 = data.drop(['price_range'], axis=1)
# target: 0(low cost),1(medium cost), 2(high cost) and 3(very high cost).
y = data['price_range'].values.reshape(-1, 1)

# ram of mobile phone - price_range
plt.figure(figsize=(16, 8))
plt.scatter(data['ram'], data['price_range'], c='black')
plt.xlabel(" ram affect cell phone sales")
plt.ylabel("price_range")
plt.show()

# weight of mobile phone - price_range
plt.figure(figsize=(16, 8))
plt.scatter(data['mobile_wt'], data['price_range'], c='black')
plt.xlabel(" mobile_wt affect cell phone sales")
plt.ylabel("price_range")
plt.show()

# Linear Regression
lin_reg = LinearRegression()
MSE5 = cross_val_score(lin_reg, x5, y, scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSE5)
print(mean_MSE)

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

# Ridge Regression
ridge = Ridge()
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(x5, y)
# find the best parameter and the best MSE
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

# Lasso Regression
lasso = Lasso()
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(x5, y)
# find the best parameter and the best MSE
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

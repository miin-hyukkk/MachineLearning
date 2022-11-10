from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("phone_sale.csv")
data = data.dropna()
x5 = data.drop(['price_range'], axis=1)
y = data['price_range'].values.reshape(-1, 1)

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

elastic_net = ElasticNet()

elastic_net_reg = GridSearchCV(elastic_net, parameters, scoring='neg_mean_squared_error', cv=5)
elastic_net_reg.fit(x5, y)
# find the best parameter and the best MSE
print(elastic_net_reg.best_params_)
print(elastic_net_reg.best_score_)

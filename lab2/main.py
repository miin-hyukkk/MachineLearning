import warnings
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MeanShift
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pyclustering.cluster.clarans import clarans
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

scalerList = [[StandardScaler(), 'standard scaler'], [MinMaxScaler(), 'minmax scaler'],
              [RobustScaler(), 'robust scaler'], [MaxAbsScaler(), 'maxAbs scaler']]
encoderList = [[OneHotEncoder(), 'one hot encoder'], [LabelEncoder(), 'label encoder'],
               [OrdinalEncoder(), 'ordinal encoder']]
metricList = ['euclidean', 'manhattan']

initParamList = ['kmeans', 'k-means++', 'random', 'random_from_data']

algorithmList = ['lloyd', 'elkan', 'auto', 'full']
DBSCANalgorithmList = ['ball_tree', 'kd_tree', 'brute']


def cl_CLARANS(X, encoder, scaler):
    data = X.copy()
    # Encode the dataset
    data['ocean_proximity'] = encoder.fit_transform(data.loc[:, ['ocean_proximity']])
    columns = data.columns
    # Scale the dataset
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(12, 12))
    for i in range(2, 12):
        clarans_instance = clarans(data=data.tolist(), number_clusters=i, numlocal=2, maxneighbor=3)
        clusters = clarans_instance.get_clusters();
        medoids = clarans_instance.get_medoids();
        score = silhouette_score(data, medoids)
        print("silhouette score when number_clusters == %d : %f" % (i, score))
        plt.subplot(4, 3, i - 1)
        plt.tight_layout()
        plt.title("number_clusters values = {}".format(i))
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.scatter(data['longitude'], data['latitude'], c=medoids, cmap=get_cmap('plasma'))
    plt.show()


def cl_Spectral(X, encoder, scaler):
    data = X.copy()
    # Encode the dataset
    data['ocean_proximity'] = encoder.fit_transform(data.loc[:, ['ocean_proximity']])
    columns = data.columns
    # Scale the dataset
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns)

    # draws scatter plot for each k value, can see visualized result of cluster and can check silhouette score of
    # clustering
    plt.figure(figsize=(12, 12))
    for i in range(2, 5):
        estimator = SpectralClustering(n_clusters=i)
        ids = estimator.fit_predict(data)
        score = silhouette_score(data, ids)
        print("silhouette score when n_components == %d : %f" % (i, score))
        plt.subplot(4, 3, i - 1)
        plt.tight_layout()
        plt.title("n_clusters values = {}".format(i))
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.scatter(data['longitude'], data['latitude'], c=ids, cmap=get_cmap('plasma'))
    plt.show()

def get_BestCombination(data):
    resultList = []
    scalerList = [[StandardScaler(), 'standard scaler'], [MinMaxScaler(), 'minmax scaler'],
                  [RobustScaler(), 'robust scaler'], [MaxAbsScaler(), 'maxAbs scaler']]
    encoderList = [[LabelEncoder(), 'label encoder'],
                   [OrdinalEncoder(), 'ordinal encoder']]
    for i, scaler in scalerList:
        for k, encoder in encoderList:
            tempResult = cl_KMeans(data,i, k)  # score,algo,k
            tempResult.append(scaler)  # 3
            tempResult.append(encoder)  # 4
            tempResult.append('Kmeans')  # 5
            resultList.append(tempResult)
    for i, scaler in scalerList:
        for k, encoder in encoderList:
            tempResult = cl_DBSCAN(data, i, k)  # score,algo,k
            tempResult.append(scaler)  # 3
            tempResult.append(encoder)  # 4
            tempResult.append('DBSCAN')  # 5
            resultList.append(tempResult)
    for i, scaler in scalerList:
        for k, encoder in encoderList:
            tempResult = cl_DBSCAN(data,i, k)  # score,algo,k
            tempResult.append(scaler)  # 3
            tempResult.append(encoder)  # 4
            tempResult.append('EM')  # 5
            resultList.append(tempResult)
    for i, scaler in scalerList:
        for k, encoder in encoderList:
            tempResult = cl_MeanShift(data,i, k)  # score
            tempResult.append(scaler)  # 3
            tempResult.append(encoder)  # 4
            tempResult.append('EM')  # 5
            resultList.append(tempResult)

    resultList.sort(key=lambda i: i[0], reverse=True)

    for i in range(10):
        print(resultList[i])


def cl_KMeans(data, scaler, encoder):
    algorithmList = ['lloyd', 'elkan', 'auto', 'full']

    data = data.copy()
    data['ocean_proximity'] = encoder.fit_transform(data[['ocean_proximity']])
    columns = data.columns
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns)
    plt.figure(figsize=(12, 12))
    for a in range(2, 12):
        for algo in algorithmList:
            estimator = KMeans(n_clusters=a, algorithm=algo)
            s = estimator.fit_predict(data)
            score = silhouette_score(data, s)
            print("silhouette score when k == %d algorithm =%s : %f" % (a, algo, score))
            plt.subplot(4, 3, a - 1)
            plt.tight_layout()
            plt.title("k values = {}".format(a))
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.scatter(data['longitude'], data['latitude'], c=s, cmap=get_cmap('plasma'))
            return [score, algo, a]
    plt.show()


def cl_DBSCAN(data, scaler, encoder):
    metricList = ['euclidean', 'manhattan']
    DBSCANalgorithmList = ['ball_tree', 'kd_tree', 'brute']

    data = data.copy()
    data['ocean_proximity'] = encoder.fit_transform(data[['ocean_proximity']])
    columns = data.columns
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns)
    plt.figure(figsize=(12, 12))
    for met in metricList:
        for algo in DBSCANalgorithmList:
            estimator = DBSCAN(eps=0.4, metric=met, algorithm=algo)
            s = estimator.fit_predict(data)
            score = silhouette_score(data, s)
            print("silhouette score when eps == %f metric ==%s algorithm ==%s: %f" % (0.4, met, algo, score))
            plt.subplot(4, 3, 1)
            plt.tight_layout()
            plt.title("eps values = {}".format(0.4))
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.scatter(data['longitude'], data['latitude'], c=s, cmap=get_cmap('plasma'))
            return [score, algo, 0.4, met]
    plt.show()


def cl_EM(data, scaler, encoder):
    initParamList = ['kmeans', 'k-means++', 'random', 'random_from_data']

    data = data.copy()
    data['ocean_proximity'] = encoder.fit_transform(data[['ocean_proximity']])
    columns = data.columns
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns)
    plt.figure(figsize=(12, 12))
    for i in range(2, 12):
        for params in initParamList:
            estimator = GaussianMixture(n_components=i, init_params=params)
            s = estimator.fit_predict(data)
            score = silhouette_score(data, s)
            print("silhouette score when n_components == %d init_params == %s : %f" % (i, params, score))
            plt.subplot(4, 3, i - 1)
            plt.tight_layout()
            plt.title("n_components values = {}".format(i))
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.scatter(data['longitude'], data['latitude'], c=s, cmap=get_cmap('plasma'))
    plt.show()
    return [score, params, i]

def cl_MeanShift(data, scaler, encoder):
    data = data.copy()
    data['ocean_proximity'] = encoder.fit_transform(data[['ocean_proximity']])
    columns = data.columns
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns)
    plt.figure(figsize=(12, 12))
    for a in range(4, 8):
            estimator = MeanShift(bandwidth=a*0.5)
            s = estimator.fit_predict(data)
            score = silhouette_score(data, s)
            print("silhouette score when bandwidth == %d: %f" % (a*0.5, score))
            plt.subplot(4, 3, a - 1)
            plt.tight_layout()
            plt.title("k values = {}".format(a))
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.scatter(data['longitude'], data['latitude'], c=s, cmap=get_cmap('plasma'))
    plt.show()


# using all attribute
new_df = pd.read_csv('housing.csv', encoding='unicode_escape')
new_df.drop(labels='median_house_value', axis=1, inplace=True)
new_df = new_df.dropna(axis=0)
new_df = new_df.dropna(axis=1)


# Using subset

# Correlation matrix for selecting attributes
plt.figure(figsize=(10, 10))
cor_mat = sns.heatmap(new_df.drop('ocean_proximity', axis=1).corr(), vmin=-1, vmax=1, annot=True)
cor_mat.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)

# housing_median_age, total_rooms, median_income  selected
sub_df = new_df.copy()
sub_df.drop(['total_bedrooms','population','households'], axis=1, inplace=True)


#타겟값으로 클러스터링한거랑 비교
# df = pd.read_csv('housing.csv', encoding='unicode_escape')
# df = df.dropna()
# df = df.drop(['ocean_proximity'], axis=1)
# df = df.astype(int)
# n_df = df['median_house_value']
# sort_df = n_df.sort_values(ascending=True)
# n = 20433
# q1 = int((0.33 * (n + 1)) - 1)
# q3 = int((0.66 * (n + 1)) - 1)
# df["target"] = np.nan
#
#
# ## I divided the 'median_house_value' data into three level
# df.loc[df['median_house_value'] < 133200, 'target'] = 'cheap'
# df.loc[df['target'] != ('cheap'), 'target'] = 'normal'
# df.loc[df['median_house_value'] > 228500, 'target'] = 'expensive'
# df = df.drop(['median_house_value'], axis=1)
# df.drop(['housing_median_age','total_rooms','median_income','total_bedrooms', 'population', 'households'], axis=1, inplace=True)
# print(df.columns)

df1 = pd.read_csv('housing.csv', encoding='unicode_escape')

df1.drop(['housing_median_age','total_rooms','median_income','total_bedrooms', 'population', 'households','ocean_proximity'], axis=1, inplace=True)

print(df1)
plt.figure(figsize=(12, 12))
for a in range(2, 6):
            estimator = KMeans(n_clusters=a)
            s = estimator.fit_predict(df1)
            score = silhouette_score(df1, s)
            print("silhouette score when k == %d: %f" % (a, score))
            plt.subplot(4, 3, a - 1)
            plt.tight_layout()
            plt.title("k values = {}".format(a))
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.scatter(df1['longitude'], df1['latitude'], c=s, cmap=get_cmap('plasma'))
plt.show()

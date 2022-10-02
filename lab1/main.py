import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)


#
# def scaling(df, scaler):
#     temp = df
#     scaled_df = temp
#     scaled_df = scaler.fit_transform(scaled_df)
#     scaled_df = pd.DataFrame(scaled_df, columns=temp.columns)
#     print(scaled_df)


# Data split

class do_classification:
    def __init__(self, df):
        self.df = df

    def get_BestCombination(self):
        resultList = []
        scalerList = [[StandardScaler(), 'standard scaler'], [MinMaxScaler(), 'minmax scaler'],
                      [RobustScaler(), 'robust scaler'], [MaxAbsScaler(), 'maxAbs scaler']]
        for i, scaler in scalerList:
            for k in range(3):
                tempResult = self.decisiontree_classifier(i, (k + 1) * 0.1)
                tempResult.append(scaler)
                tempResult.append('decision tree')
                tempResult.append((k + 1) * 0.1)
                resultList.append(tempResult)

        for i, scaler in scalerList:
            for k in range(3):
                tempResult = self.logistic_regression(i, (k + 1) * 0.1)
                tempResult.append(scaler)
                tempResult.append('logistic regression')
                tempResult.append((k + 1) * 0.1)
                resultList.append(tempResult)

        for i, scaler in scalerList:
            for k in range(3):
                tempResult = self.SVM(i, (k + 1) * 0.1)
                tempResult.append(scaler)
                tempResult.append('support vector machine')
                tempResult.append((k + 1) * 0.1)
                resultList.append(tempResult)

        resultList.sort(key=lambda i: i[0], reverse=True)

        for i in range(5):
            print("%d combination - score : %f model : %s scaler %s parameter : %s data split %s" % (
                i + 1, resultList[i][0], resultList[i][4], resultList[i][2], resultList[i][1], resultList[i][3]))

    def decisiontree_classifier(self, scaler, split):
        temp = self.df.copy()
        temp.iloc[:, :8] = scaler.fit_transform(temp.iloc[:, :8])
        dtc = DecisionTreeClassifier()

        grid_param = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                      'max_depth': [2, 4, 6, 8, 10, 12, 14, 16], 'min_samples_split': [2, 3, 4, 5, 6, 7]}
        model = GridSearchCV(estimator=dtc, param_grid=grid_param, cv=5, n_jobs=-1)
        # print(temp)
        Y = temp.iloc[:, 8]
        data = temp.iloc[:, :8]

        data_train, data_test, target_train, target_test = train_test_split(data, Y, test_size=split,
                                                                            random_state=42)

        model.fit(data_train, target_train)

        print('GridsearchCV hyperparameter: ', model.best_params_)
        print('GridsearchCV best accuracy: {0:.4f}'.format(model.best_score_))

        return [model.best_score_, model.best_params_]

    def logistic_regression(self, scaler, split):
        temp = self.df.copy()
        temp.iloc[:, :8] = scaler.fit_transform(temp.iloc[:, :8])
        lgr = LogisticRegression()

        grid_param = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'max_iter': [50, 100, 200, 400, 600, 800, 1000], 'warm_start': [True, False]}
        model = GridSearchCV(estimator=lgr, param_grid=grid_param, cv=5, n_jobs=-1)

        Y = temp.iloc[:, 8]
        data = temp.iloc[:, :8]

        data_train, data_test, target_train, target_test = train_test_split(data, Y, test_size=split,
                                                                            random_state=42)

        model.fit(data_train, target_train)

        print('GridsearchCV hyperparameter: ', model.best_params_)
        print('GridsearchCV best accuracy: {0:.4f}'.format(model.best_score_))

        return [model.best_score_, model.best_params_]

    def SVM(self, scaler, split):
        temp = self.df.copy()
        temp.iloc[:, :8] = scaler.fit_transform(temp.iloc[:, :8])
        svm_clf = SVC()
        grid_param = {'kernel': ['poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto'],
                      'C': [0.01,0.1,1]}
        model = GridSearchCV(estimator=svm_clf, param_grid=grid_param, cv=5, n_jobs=-1)
        Y = temp.iloc[:, 8]
        data = temp.iloc[:, :8]
        data_train, data_test, target_train, target_test = train_test_split(data, Y, test_size=split,
                                                                            random_state=42)
        model.fit(data_train, target_train)

        print('GridsearchCV hyperparameter: ', model.best_params_)
        print('GridsearchCV best accuracy: {0:.4f}'.format(model.best_score_))

        return [model.best_score_, model.best_params_]


new_df = pd.read_csv('breast-cancer-wisconsin.csv', encoding='unicode_escape')
new_df.drop(labels='Bare Nuclei', axis=1, inplace=True)
new_df.drop(labels='ID', axis=1, inplace=True)

a = do_classification(new_df)
a.get_BestCombination()


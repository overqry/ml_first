import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv('../data/set_1.csv', encoding='windows-1251', delimiter=';')
data['BuilderObjectRu'] = data['BuilderObjectRu'].fillna('Тилимилитрямдия')

data['plan_timestamp'] = data['BuildFinishDate'].apply(
    lambda it: dt.datetime.strptime(str(it), '%d.%m.%Y %H:%M').timestamp() if str(it) != 'nan'
    else dt.datetime.strptime('01.01.2900','%d.%m.%Y').timestamp())

data['changes_timestamp'] = data['PDChangesBuildFinishDate'].apply(
    lambda it: dt.datetime.strptime(str(it), '%d.%m.%Y %H:%M').timestamp() if str(it) != 'nan'
    else dt.datetime.strptime('01.01.3000','%d.%m.%Y').timestamp())

data['diff_changes_plan'] = data['changes_timestamp'] - data['plan_timestamp']

year_timestamp = dt.datetime.strptime('01.01.2901','%d.%m.%Y').timestamp() - dt.datetime.strptime('01.01.2900','%d.%m.%Y').timestamp()
data['is_deviation'] = data['diff_changes_plan'].apply(
    lambda it: 1 if it > year_timestamp else 0
)

data_describe = pd.DataFrame(data.describe(include='object'))
data_corr = pd.DataFrame(data.corr(numeric_only=True))

X = data.drop(['BuilderCompany', 'BuilderCompanyCode', 'BuilderObjectRu', 'BuildFinishDate', 'PDChangesBuildFinishDate', 'is_deviation', 'plan_timestamp', 'changes_timestamp'], axis=1)
y = data['is_deviation']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

column_transformer = ColumnTransformer([
    ('scaling', StandardScaler(), X.columns)
])
X_transformed = column_transformer.fit_transform(Xtrain)
X_test_transformed = column_transformer.fit_transform(Xtest)

train_set = Pool(X_transformed, ytrain)
test_set = Pool(X_test_transformed, ytest)

gbm = CatBoostClassifier(
    iterations = 100,
    depth = 3,
    learning_rate = 0.1,
    loss_function = 'Logloss',
    eval_metric = 'AUC',
    verbose = False)

gbm.fit(train_set, eval_set = test_set)

eval_metrics = gbm.get_evals_result()
plt.plot(eval_metrics['validation']['AUC'])
plt.xlabel('n_trees')
plt.ylabel('AUC')
plt.grid()

gbm_pred = gbm.predict(X_test_transformed)
accuracy = accuracy_score(ytest, gbm_pred)

gbm_pred_proba = gbm.predict_proba(X_test_transformed)
classes = (gbm_pred_proba[:, 1] > 0.5)
recall = recall_score(ytest, classes)
precision = precision_score(ytest, classes)
aucroc = roc_auc_score(ytest,  gbm_pred_proba[:,1])

print("accuracy: ", accuracy)
print("recall: ", recall)
print("precision: ", precision)
print("aucroc: ", aucroc)

coef = pd.DataFrame({'features' : list(X.columns), 'importances' : list(gbm.get_feature_importance())})
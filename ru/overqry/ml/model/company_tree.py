# нужно получить компанию, у которой объекты застройки сдаются быстрее всех

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

data = pd.read_csv('../data/set_1.csv', encoding='windows-1251', delimiter=';')

data = data.drop(columns='BuilderCompanyCode', axis=1)
data['BuilderObjectRu'] = data['BuilderObjectRu'].fillna('Тилимилитрямдия')

data['is_plan_success'] = data['BuildFinishDate'].apply(
    lambda it: 1 if str(it) != 'nan'
                    and dt.datetime.now() >= dt.datetime.strptime(str(it), '%d.%m.%Y %H:%M:%S')
    else 0)

data['plan_timestamp'] = data['BuildFinishDate'].apply(
    lambda it: dt.datetime.strptime(str(it), '%d.%m.%Y %H:%M:%S').timestamp() if str(it) != 'nan'
    else dt.datetime.strptime('01.01.2900','%d.%m.%Y').timestamp())

data['changes_timestamp'] = data['PDChangesBuildFinishDate'].apply(
    lambda it: dt.datetime.strptime(str(it), '%d.%m.%Y %H:%M:%S').timestamp() if str(it) != 'nan'
    else dt.datetime.strptime('01.01.3000','%d.%m.%Y').timestamp())

data['plan_timestamp'] = data['plan_timestamp'].apply(
    lambda it: 0 if dt.datetime.now().timestamp() >= it else int(it))

data['changes_timestamp'] = data['changes_timestamp'].apply(
    lambda it: 0 if dt.datetime.now().timestamp() >= it else int(it))

data['diff_changes_plan'] = data['changes_timestamp'] - data['plan_timestamp']

data_describe = pd.DataFrame(data.describe(include='object'))
data_corr = pd.DataFrame(data.corr(numeric_only=True))

X = data.drop(['BuilderCompany','BuilderObjectRu','BuildFinishDate', 'PDChangesBuildFinishDate','is_plan_success', 'plan_timestamp', 'changes_timestamp'], axis=1)
y = data.is_plan_success

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

ss = StandardScaler()
ss.fit(Xtrain)
Xtrain = pd.DataFrame(ss.transform(Xtrain), columns=X.columns)
Xtest = pd.DataFrame(ss.transform(Xtest), columns=X.columns)

train_set = Pool(Xtrain, ytrain)
test_set = Pool(Xtest, ytest)

gbm = CatBoostClassifier(
    iterations = 300,
    depth = 2,
    learning_rate = 0.1,
    loss_function = 'Logloss',
    eval_metric = 'AUC',
    verbose = False)

gbm.fit(train_set, eval_set = test_set)

gbm_preds = gbm.predict(Xtest)

eval_metrics = gbm.get_evals_result()
plt.plot(eval_metrics['validation']['AUC'])
plt.xlabel('n_trees')
plt.ylabel('AUC')
plt.grid()

recall = recall_score(ytest, gbm_preds)
precision = precision_score(ytest, gbm_preds)
aucroc = roc_auc_score(ytest,  gbm_preds)


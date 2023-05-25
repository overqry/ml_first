# нужно получить компанию, у которой объекты застройки сдаются быстрее всех

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

#пока бесполезно, но вывел в IDEA графики через плагины
#plt.scatter(data.plan_timestamp, data.diff_changes_plan, c=data.diff_changes_plan)

#TODO надо обдумать
#print(data.groupby('diff_changes_plan')['BuilderCompany'].mean())
#data[data.diff_changes_plan == 0]['is_plan_success'].hist(bins=30)
#data[data.diff_changes_plan != 0]['is_plan_success'].hist(bins=30)

X = data.drop(['BuilderCompany','BuilderObjectRu','BuildFinishDate', 'PDChangesBuildFinishDate','diff_changes_plan', 'plan_timestamp', 'changes_timestamp'], axis=1)
y = data.diff_changes_plan

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression()
model.fit(Xtrain, ytrain)
ypred_test = model.predict(Xtest)
print(accuracy_score(ytest, ypred_test))
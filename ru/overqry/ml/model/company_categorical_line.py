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

X_full = data.copy().drop(['BuilderCompanyCode', 'BuilderObjectRu', 'BuildFinishDate', 'PDChangesBuildFinishDate','is_deviation', 'plan_timestamp', 'changes_timestamp'], axis=1)
X = data.drop(['BuilderCompany', 'BuilderCompanyCode', 'BuilderObjectRu', 'BuildFinishDate', 'PDChangesBuildFinishDate','is_deviation', 'plan_timestamp', 'changes_timestamp'], axis=1)
y = data['is_deviation']
#Xtrain, Xtest, ytrain, ytest = train_test_split(X_full, y, test_size=0.25, random_state=42)

categorical = ['BuilderCompany']
numeric_features = X.columns
column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('scaling', StandardScaler(), numeric_features)
])
X_transformed = column_transformer.fit_transform(X_full)
Xtrain, Xtest, ytrain, ytest = train_test_split(X_transformed, y, test_size=0.25, random_state=42)
#X_test_transformed = column_transformer.fit_transform(Xtest)

model = LogisticRegression()
model.fit(Xtrain, ytrain)

pred = model.predict(Xtest)
accuracy = accuracy_score(ytest, pred)

pred_proba = model.predict_proba(Xtest)
classes = (pred_proba[:, 0] > 0.1)
recall = recall_score(ytest, classes)
precision = precision_score(ytest, classes)
aucroc = roc_auc_score(ytest,  pred_proba[:,1])

print("accuracy: ", accuracy)
print("recall: ", recall)
print("precision: ", precision)
print("aucroc: ", aucroc)

# нужно получить компанию, у которой объекты застройки сдаются быстрее всех

import datetime as dt

import pandas as pd

data = pd.read_csv('../data/set_1.csv', encoding='windows-1251', delimiter=';')

print(data.isna().sum())
print(data.dtypes)
print(data.BuilderObjectRu.value_counts(dropna=False))

data = data.drop(columns='BuilderCompanyCode', axis=1)
data['BuilderObjectRu'] = data['BuilderObjectRu'].fillna('Тилимилитрямдия')

# успешное строительство
data['is_finish_success'] = data['BuildFinishDate'].apply(
    lambda it: 1 if str(it) != 'nan'
                    and data['PDChangesBuildFinishDate'].isnull
                    and dt.datetime.now() >= dt.datetime.strptime(str(it), '%d.%m.%Y %H:%M:%S')
    else 0)

# всегда есть актуальная информация по предполагаемой сдаче объекта
data['is_changes_success'] = data['PDChangesBuildFinishDate'].apply(
    lambda it: 1 if str(it) != 'nan'
                    and dt.datetime.now() >= dt.datetime.strptime(str(it), '%d.%m.%Y %H:%M:%S')
                    and str(data['BuildFinishDate']) != 'nan'
    else 0)

# нет информации о плановой сдаче объекта
data["is_finish_failure"] = data['BuildFinishDate'].apply(lambda it: 1 if str(it) == '' else 0)

# нет информации о предполагаемой сдаче объекта, при этом она должна быть
data["is_changes_failure"] = data['PDChangesBuildFinishDate'].apply(
    lambda it: 1 if str(it) == ''
                    and data['BuildFinishDate'] != 'nan'
                    and dt.datetime.now() < dt.datetime.strptime(str(data['BuildFinishDate']), '%d.%m.%Y %H:%M:%S')
    else 0)

print(data.info())

print(data['is_finish_success'].value_counts(normalize=True))

print(data.describe())

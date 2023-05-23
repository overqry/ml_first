# нужно получить компанию, у которой объекты застройки сдаются быстрее всех

import datetime as dt

import pandas as pd

data = pd.read_csv('../data/set_1.csv', encoding='windows-1251', delimiter=';')

data = data.drop(columns='BuilderCompanyCode', axis=1)
data['BuilderObjectRu'] = data['BuilderObjectRu'].fillna('Тилимилитрямдия')

# успешное строительство
data['is_finish_success'] = data['BuildFinishDate'].apply(
    lambda it: 1 if str(it) != 'nan'
                    and dt.datetime.now() >= dt.datetime.strptime(str(it), '%d.%m.%Y %H:%M:%S')
    else 0)

# разница между плановой и предполагаемой сдачей объекта
data['diff_changes_finish'] = data['PDChangesBuildFinishDate'].apply(lambda changes:
                                 dt.datetime.strptime(str(changes), '%d.%m.%Y %H:%M:%S').timestamp()
                                 if str(changes) != 'nan'
                                    and dt.datetime.now() <= dt.datetime.strptime(str(changes), '%d.%m.%Y %H:%M:%S')
                                 else dt.datetime.strptime('01.01.3000', '%d.%m.%Y').timestamp()) - \
                              data['BuildFinishDate'].apply(lambda finish:
                                dt.datetime.strptime(str(finish), '%d.%m.%Y %H:%M:%S').timestamp()
                                if str(finish) != 'nan'
                                   and dt.datetime.now() <= dt.datetime.strptime(str(finish), '%d.%m.%Y %H:%M:%S')
                                else dt.datetime.strptime('01.01.3000', '%d.%m.%Y').timestamp())

# нет информации о плановой сдаче объекта
data["is_finish_failure"] = data['BuildFinishDate'].apply(lambda it: 1 if str(it) == 'nan' else 0)

# нет информации о предполагаемой сдаче объекта
data["is_changes_failure"] = data['PDChangesBuildFinishDate'].apply(lambda it: 1 if str(it) == 'nan' else 0)

print(data['is_finish_success'].value_counts(normalize=True))
print(data['is_finish_failure'].value_counts(normalize=True))
print(data['is_changes_failure'].value_counts(normalize=True))



print(data['diff_changes_finish'].value_counts(normalize=True))

#print(data.describe())

#Klasyfikacja zachowań
#Anna Wróbel Informatyka i Ekonometria


# Bliblioteki
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

pd.set_option('float_format', '{:.3f}'.format)

#wczytanie danych
data = pd.read_csv('C:/MachineLearning/data.csv', header = None)
print(data)

data[data.isin(['unknown'])].any(axis = 1)

data = pd.read_csv('C:/ML/data.csv', header=None, sep=';',  names = ['age', 'job', 'marital', 'education','default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'result'
], decimal='.', na_values='unknown',dtype={'job':'category', 'marital':'category', 'education':'category',	'default':'category', 'housing':'category', 'loan':'category', 'contact':'category', 'month':'category', 'day_of_week':'category',  'previous':'category', 'poutcome':'category',  'result':'category'})

#określenie rozmiaru zbioru
print(str(data.shape[0]) + ' wierszy.')
print(str(data.shape[1]) + ' kolumn.')
print(data)

#Przygotowanie danych
summary=pd.DataFrame(data.dtypes, columns=['Dtype'])
summary['Nulls'] = pd.DataFrame(data.isnull().any())
summary['Sum_of_nulls'] = pd.DataFrame(data.isnull().sum())
summary['Per_of_nulls'] = round((data.apply(pd.isnull).mean()*100),2)
summary.Dtype = summary.Dtype.astype(str)
summary

print(summary)

print(str(round(data.isnull().any(axis=1).sum()/data.shape[0]*100,2))+'% obserwacji zawiera braki w danych.')

#25,98% obserwacji zaweira braki w danych

#obserwacje zawierajace braki w danych zostaja usuniete
data.dropna(inplace=True)
print('Pozostało ' + str(data.shape[0]) + ' wierszy.')


#pozostało 30488 wierszy

unique = data.select_dtypes(exclude=['float','int']).describe()
print(unique)


#zamiana wartosci zmiennych

data.default.replace(['yes','no'],[1,0], inplace=True)
data.housing.replace(['yes','no'],[1,0], inplace=True)
data.loan.replace(['yes','no'],[1,0], inplace=True)
data.contact.replace(['telephone','cellular'],[1,0], inplace=True)
data.result.replace(['yes','no'],[1,0], inplace=True)

data['default'] = data['default'].astype('uint8')
data['housing'] = data['housing'].astype('uint8')
data['loan'] = data['loan'].astype('uint8')
data['contact'] = data['contact'].astype('uint8')
data['result'] = data['result'].astype('uint8')

#Analiza danych
#zmienne numeryczne
stats = data.select_dtypes(['float', 'int']).describe()
stats = stats.transpose()
stats = stats[['count','std','min','25%','50%','75%','max','mean']]
print(stats)

#zmienne kategoryczne i binarne
categ=data.select_dtypes(['category']).describe()
print(categ)

x =data.select_dtypes(['category', 'uint8']).columns
print(x)

#job

cat1 = pd.DataFrame(data['job'].value_counts())
cat1.rename(columns={'job':'Num_of_obs'}, inplace=True)
cat1['Per_of_obs'] = cat1['Num_of_obs']/data.shape[0]*100
print(cat1)

cat1_result = pd.crosstab(data.job, data.result)
print(cat1_result)

#martial
cat2 = pd.DataFrame(data['marital'].value_counts())
cat2.rename(columns={'marital':'Num_of_obs'}, inplace=True)
cat2['Per_of_obs'] = cat1['Num_of_obs']/data.shape[0]*100
print(cat2)

cat2_result = pd.crosstab(data.marital, data.result)
print(cat2_result)

#education
cat3 = pd.DataFrame(data['education'].value_counts())
cat3.rename(columns={'education':'Num_of_obs'}, inplace=True)
cat3['Per_of_obs'] = cat3['Num_of_obs']/data.shape[0]*100
print(cat3)

cat3_result = pd.crosstab(data.education, data.result)
print(cat3_result)

#month
cat4 = pd.DataFrame(data['month'].value_counts())
cat4.rename(columns={'month':'Num_of_obs'}, inplace=True)
cat4['Per_of_obs'] = cat4['Num_of_obs']/data.shape[0]*100
print(cat4)

cat4_result= pd.crosstab(data.month, data.result)
print(cat4_result)

#day_of_week

cat5 = pd.DataFrame(data['day_of_week'].value_counts())
cat5.rename(columns={'day_of_week':'Num_of_obs'}, inplace=True)
cat5['Per_of_obs'] = cat5['Num_of_obs']/data.shape[0]*100
print(cat5)

cat5_result = pd.crosstab(data.day_of_week, data.result)
print(cat5_result)

#previous

cat6 = pd.DataFrame(data['previous'].value_counts())
cat6.rename(columns={'previous':'Num_of_obs'}, inplace=True)
cat6['Per_of_obs'] = cat6['Num_of_obs']/data.shape[0]*100
print(cat6)

cat6_result = pd.crosstab(data.previous, data.result)
print(cat6_result)

#poutcome
cat7 = pd.DataFrame(data['poutcome'].value_counts())
cat7.rename(columns={'poutcome':'Num_of_obs'}, inplace=True)
cat7['Per_of_obs'] = cat7['Num_of_obs']/data.shape[0]*100
print(cat7)

cat7_result = pd.crosstab(data.poutcome, data.result)
print(cat7_result)

#usunięcie zmiennych
data.drop(['marital','previous','poutcome'], axis = 1, inplace = True)

#Zmiana kodowania zmiennych
a = data.select_dtypes(include=['category']).describe()
print(a)

data = pd.concat([data, pd.get_dummies(data.job, prefix='job__')], axis = 1)
data = pd.concat([data,pd.get_dummies(data.education, prefix='education__')], axis = 1)
data = pd.concat([data,pd.get_dummies(data.month, prefix='month__')], axis = 1)
data = pd.concat([data,pd.get_dummies(data.day_of_week, prefix='day_of_week__')], axis = 1)

data.drop(['job', 'education', 'month', 'day_of_week',], axis = 1, inplace = True)

#podział zbioru
part = data.result.value_counts(normalize=True)
print(part)

y=data.result
data.drop('result',axis=1,inplace=True)



x_tr, x_te, y_tr, y_te = train_test_split(data, y, test_size = 0.2, random_state = 7042018, stratify = y)


print(y_tr.value_counts(normalize = True))
print(y_te.value_counts(normalize = True))

#Budowa modelu 1


model = DecisionTreeClassifier()


cv = cross_val_score(model, x_tr, y_tr, cv = 10, scoring = 'accuracy')


print('Średnie Accuracy: ' + str(cv.mean().round(3)))
print('Stabilność: ' + str((cv.std()*100/cv.mean()).round(3)) + '%')



w = x_tr.columns.shape
print(w)



model.fit(x_tr, y_tr)
pred = model.predict(x_te)
print( str(round(accuracy_score(pred, y_te),3)))


#Budowa modelu 2
selector = RFE(model, 9, 1)
cols = x_tr.iloc[:,selector.fit(x_tr, y_tr).support_].columns
print(cols)

parameters = {'criterion':('entropy', 'gini'), 'splitter':('best','random'), 'max_depth':np.arange(1, 6), 'min_samples_split':np.arange(2,10), 'min_samples_leaf':np.arange(1,5)}
classifier2 = GridSearchCV(DecisionTreeClassifier(), parameters, cv=10)
classifier2.fit(x_tr[cols], y_tr)
classifier2.best_params_

cv = cross_val_score(DecisionTreeClassifier(**classifier2.best_params_), x_tr[cols], y_tr, cv = 10, scoring = 'accuracy')
print('Średnie Accuracy: ' + str(cv.mean().round(3)))
print('Stabilność: ' + str((cv.std()*100/cv.mean()).round(3)) + '%')


model = DecisionTreeClassifier(**classifier2.best_params_)
model.fit(x_tr[cols], y_tr)
pred = model.predict(x_te[cols])

print('Ostateczny wynik: ' + str(round(accuracy_score(pred, y_te),3)))

#Sprawdzenie przykładowych decyzji
samples = x_te[cols].sample(10).sort_index()
print(samples)
print(y_te[y_te.index.isin(samples.index)].sort_index())
print(model.predict(samples))



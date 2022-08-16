# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:42:18 2022

@author: Prashant
"""

import os
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz




path= "C:/Users/prash/Downloads/"
filename="KSI.csv"
fullpath=os.path.join(path, filename)

df=pd.read_csv(fullpath)
df.dtypes
df.isnull().any()
print(df.describe())


df = df.replace('<Null>',np.NaN).astype(df.dtypes)
df = df.replace('unknown',np.NaN).astype(df.dtypes)
df.isnull().any().count
df.isnull().sum()

df= df.drop(['X', 'Y', 'WARDNUM', 'DIVISION','POLICE_DIVISION','OFFSET','ACCNUM','INDEX_','FATAL_NO', 'PEDTYPE', 'PEDACT','PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND','ObjectId', 'HOOD_ID', 'NEIGHBOURHOOD', 'MANOEUVER', 'INITDIR', 'VEHTYPE', 'DRIVACT','DRIVCOND'], axis = 1)

df.ACCLASS.value_counts()
df.ACCLASS.replace(to_replace = ['Property Damage Only', 'Non-Fatal Injury'], value = 'Non-Fatal', inplace = True)
df.DISTRICT.replace(to_replace = 'Toronto East York', value = 'Toronto and East York', inplace = True)
df.VISIBILITY.replace(to_replace=['Other'], value=np.nan,inplace=True)
df.RDSFCOND.replace(to_replace=['Other'], value=np.nan,inplace=True)
df.IMPACTYPE.replace(to_replace=['Other'], value=np.nan,inplace=True)


df.ACCLASS.value_counts()
le = LabelEncoder()
label = le.fit_transform(df['ACCLASS'])
df.drop("ACCLASS", axis=1, inplace=True)
df["ACCLASS"] = label

null_columns=['STREET2','ROAD_CLASS', 'DISTRICT', 'LOCCOORD','TRAFFCTL','VISIBILITY','RDSFCOND','IMPACTYPE', 'INVAGE', 'INJURY']

#df['WARDNUM']=df['WARDNUM'].fillna(0).astype(str).str.extractall(r'(^(?:\d+)?[^,])').unstack().astype(int)
#similarly, some 'division' cells have more than one values separated by cells; choosing the first one and ignoring the rest
#df['DIVISION']=df['DIVISION'].fillna(0).astype(str).str.extractall(r'(^(?:\d+)?[^,])').unstack().astype(int)

df['DATE'] = pd.to_datetime(df['DATE'], format = '%Y/%m/%d %H:%M:%S')

df.insert(1, 'MONTH', df['DATE'].dt.month)

df.insert(2, 'DAY', df['DATE'].dt.day)

df.drop(['YEAR', 'DATE', 'HOUR'], axis = 1, inplace = True)
 
mappable=['PEDESTRIAN','CYCLIST','AUTOMOBILE','MOTORCYCLE', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER','SPEEDING','AG_DRIV', 'DISABILITY','REDLIGHT', 'TRUCK', 'ALCOHOL']

for col in mappable:
    df[col]=df[col].map({'Yes':1, np.nan:0})
    
print(df['PEDESTRIAN'])

df['LOCCOORD'].fillna(df.ACCLOC[df['LOCCOORD'].isna()], inplace=True)

df.drop(['ACCLOC'], axis=1, inplace=True)

print(df.isnull().sum())
df_features=['MONTH','DAY','TIME','STREET1','STREET2', 'LATITUDE', 'LONGITUDE', 'ROAD_CLASS', 'DISTRICT', 'LOCCOORD', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE','INVAGE', 'INJURY', 'AUTOMOBILE', 'AG_DRIV', 'PEDESTRIAN', 'CYCLIST', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH','EMERG_VEH','PASSENGER','SPEEDING','DISABILITY', 'ALCOHOL', 'REDLIGHT']
numerical_features=['MONTH', 'DAY','TIME', 'LATITUDE', 'LONGITUDE']
categorical_features=['STREET1', 'STREET2','ROAD_CLASS', 'DISTRICT', 'LOCCOORD', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE','INVAGE', 'INJURY']

sns.heatmap(df[mappable].corr())
#replacing null values with most frequent instances of that class
#df[null_columns]=df[null_columns].fillna(df.groupby('ACCLASS')[null_columns].transform(lambda x:x.mode().iat[0]))


X_train, X_test, Y_train, Y_test=train_test_split(df[df_features], df['ACCLASS'],test_size=0.2, random_state=57, stratify=df['ACCLASS'])


num_pipeline = Pipeline(
    [ ("scaler", StandardScaler())]
)
     
categoricalimputer=SimpleImputer( strategy="most_frequent")

cat_pipeline = Pipeline(
    [("imputer", categoricalimputer),
    ("encoder", OneHotEncoder( handle_unknown='ignore'))]
  )



full_pipeline = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features),
    ], remainder='passthrough')

df.isnull().sum()




#Logistic Regression: Prashant

pipe_lr=Pipeline([('full', full_pipeline),
                            ('lr', LogisticRegression())])
param_lr = {'lr__penalty': ['l1', 'l2', 'elasticnet'],
              'lr__C': [0.01, 0.1, 1, 10, 100],
              'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']}

grid_search_lr=GridSearchCV(estimator=pipe_lr, 
                          param_grid=param_lr, 
                          scoring='accuracy', 
                          refit=True, 
                          verbose=0,
                          cv=3)

grid_search_lr.fit(X_train, Y_train.values.ravel())


print(grid_search_lr.best_params_)
print(grid_search_lr.best_estimator_)
print(grid_search_lr.best_score_)

best_model=grid_search_lr.best_estimator_

predictions=best_model.predict(X_test)

print(classification_report(predictions, Y_test))
print(confusion_matrix(Y_test, predictions))


joblib.dump(best_model, 'bestmodel_lr.pkl')
joblib.dump(pipe_lr, 'pipeline_lr.pkl')
joblib.dump(df_features, 'model_columns.pkl')




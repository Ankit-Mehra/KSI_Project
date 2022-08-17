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



import os
path = os.getcwd()
filename="Data/KSI.csv"
fullpath=os.path.join(path, filename)
print(fullpath)


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


#SVM: Ankit Mehra


# Helper Function to plot confusion Matrix
def plot_matrix(clf,X_test,y_test,model:str):
        # get prediction
        pred = clf.predict(X_test)
        confusion_mat = confusion_matrix(y_test,pred)
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        fig, axes = plt.subplots(1, figsize=(10, 5))

        group_counts = [f"{value:0.0f}" for value in
                confusion_mat.flatten()]

        group_percentages = [f"{value:.2%}" for value in
                        confusion_mat.flatten()/np.sum(confusion_mat)]

        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]

        labels = np.asarray(labels).reshape(2,2)
        #     plt.figure(figsize= (8,6))
        ax = sns.heatmap(confusion_mat, annot=labels, 
                fmt='', cmap='Blues',ax=axes)

        ax.set_title(f"Confusion Matrix for {model} Model\n\n")
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');

        # Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])

        ## Display the visualization of the Confusion Matrix.
        plt.show()




# Instantiate a default Support Vector Classifier
svc = SVC()

# Pipeline Object to streamline the process
pipeline_svc = Pipeline([
    ('full', full_pipeline),
    ('svc', svc)
    ])


pipeline_svc.fit(X_train, Y_train)

scores = cross_val_score(pipeline_svc,
                        X_train,
                        Y_train,
                        cv=10,
                        n_jobs=-1,
                        verbose=1)
print(scores)
print(scores.mean())


# Predictions
y_pred_svc = pipeline_svc.predict(X_test)

#plot confusion Matrix
plot_matrix(pipeline_svc,X_test,Y_test,"SVC")


#fine Tuning the model

# Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
param_grid = {'svc__kernel': ['linear', 'rbf', 'poly'],
              'svc__C': [0.01, 0.1, 1, 10, 100],
              'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
              'svc__degree': [2, 3]}

# Create a GridSearchCV object
grid_search_svc = GridSearchCV(estimator = pipeline_svc,
                                 param_grid = param_grid,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)

grid_search_svc.fit(X_train, Y_train)

print(grid_search_svc.best_params_)
print(grid_search_svc.best_estimator_)
print(grid_search_svc.best_score_)

best_model_svm = grid_search_svc.best_estimator_

plot_matrix(best_model,X_test,Y_test,"SVC")
joblib.dump(best_model_svm, 'Models/bestmodel_svm.pkl')
joblib.dump(pipeline_svc, 'Models/pipeline_svc.pkl')

# K-Nearest Neighbours MODEL: Ayesha

knn = KNeighborsClassifier()

pipe_knn=Pipeline([('full', full_pipeline),
                    ('knn', KNeighborsClassifier())])

param_grid_knn=[
    {'knn__n_neighbors':[3,5,11,19],
    'knn__weights':['uniform', 'distance'],
    'knn__metric': ['euclidean', 'Manhattan']}]

grid_search_knn=GridSearchCV(estimator=pipe_knn, 
                          param_grid=param_grid_knn, 
                          scoring='accuracy', 
                          refit=True, 
                          verbose=1,
                          cv=3)

print(grid_search_knn)
grid_search_knn.fit(X_train, Y_train.values)

print(f"Best Parameters: {grid_search_knn.best_params_}")   
print(f"Best Estimator: {grid_search_knn.best_estimator_}")
print(f"Best Accuracy score: {grid_search_knn.best_score_}")

# print(classification_report(grid_search_knn.predict(X_test), Y_test))

best_model=grid_search_knn.best_estimator_

predictions=best_model.predict(X_test)

print(classification_report(predictions, Y_test))
print(confusion_matrix(Y_test, predictions))

joblib.dump(best_model, 'Models/bestmodel_knn.pkl')
joblib.dump(pipe_knn, 'Models/pipeline_knn.pkl')

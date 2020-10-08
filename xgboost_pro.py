# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 01:11:10 2020

@author: Koray
"""

import pandas as pd 


data=pd.read_csv("Churn_Modelling.csv")

print(data.head())

print(data.columns)

print(data.describe())

print(data.info())

#%% Bazı Sütunların Silinmesi

data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)


#%% One Hot Encoder 

data=pd.get_dummies(data,columns=['Geography','Gender','HasCrCard','IsActiveMember'],
                    prefix=['Geography','Gender','HasCrCard','IsActiveMember'])

#%% Exited sütunun alınması

exited=data.iloc[:,6:7]

#%% Exited sütunun datamızdan silinmesi

data=data.drop(['Exited'],axis=1)

#%% Train ve Test Olarak Ayrılması

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,exited,test_size=0.33,random_state=0)

#%% Standartlaştırma

from sklearn.preprocessing import StandardScaler

s_scaler=StandardScaler()

x_train=s_scaler.fit_transform(x_train)

x_test=s_scaler.transform(x_test)

#%% Logistic Regression

from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()

log_reg.fit(x_train,y_train)

y_pred=log_reg.predict(x_test)

#%% Confusion Matrix ve accuracy score

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test, y_pred)

acc_score=accuracy_score(y_test, y_pred)*100

#%% XGBoost 

from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

xgb_pred=xgb.predict(x_test)

#%% XGBoost Confusion Matrix ve Accuracy score

cm2=confusion_matrix(y_test, xgb_pred)

xgb_acc_score=accuracy_score(y_test, xgb_pred)*100

#%% Sinir Ağı 

import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

model.add(Dense(units=64,activation=tf.nn.relu,input_dim=(15)))

model.add(Dense(units=32,activation=tf.nn.relu))

model.add(Dense(units=1,activation=tf.nn.sigmoid))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

hist=model.fit(x_train,y_train,epochs=10,verbose=1)

#%% Predict

nn_pred=model.predict(x_test)

nn_pred=nn_pred > 0.5
#%% Sinir Ağı Confusion Matrix ve accuracy score 

nn_cm=confusion_matrix(y_test, nn_pred)

nn_acc_score=accuracy_score(y_test, nn_pred)*100


#%% Sinir Ağı Seaborn 

import seaborn as sns 

nn_heatmap=sns.heatmap(nn_cm,annot=True,cbar=False,cmap='YlOrBr')

#%% Logistic Regression 

lr_heatmap=sns.heatmap(cm,annot=True,cbar=False,cmap='RdPu')

#%% XGBoost 

sns.set(font_scale=2)

xgb_heatmap=sns.heatmap(cm2, annot=True,cbar=False,cmap='YlGnBu')






















































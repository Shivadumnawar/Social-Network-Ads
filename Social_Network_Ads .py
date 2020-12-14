# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:12:11 2020

@author: shiva dumnawar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('Social_Network_Ads .csv')

df.info()

df.describe()

# check null values
df.isnull().sum()   # no null values

df.drop(['User ID'], axis=1, inplace=True)

#check outliers
df.plot(kind= 'box')  # no outliers 

df['Gender'].value_counts()
df['Purchased'].value_counts()

plt.figure(figsize=(8,6))
sns.countplot(x= 'Purchased', data= df)

plt.figure(figsize=(8,6))
sns.countplot(x= 'Purchased', hue= 'Gender', data= df)

# one hot encoding
df= pd.get_dummies(df, columns=['Gender'], drop_first=True)

df.columns

new_order= ['Age', 'Gender_Male', 'EstimatedSalary', 'Purchased']
df= df[new_order]

# correlation
plt.figure(figsize=(8,6))
c= df.corr()
sns.heatmap(c, cmap= 'coolwarm', annot= True)
plt.tight_layout()

X= df.iloc[:, :-1]
y= df.iloc[:, -1].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=72 )

# scaling
cols= ['Age', 'EstimatedSalary']

from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

X_train[cols]= ss.fit_transform(X_train[cols])
X_test[cols]= ss.fit_transform(X_test[cols])

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()

clf.fit(X_train, y_train.ravel())

pred= clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(y_test, pred))

print(accuracy_score(y_test, pred))

print(classification_report(y_test, pred))


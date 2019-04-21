# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 19:19:52 2018

@author: HP
"""
import numpy as np
#age = np.random.randint(18,60,100)
#amnt = np.random.randint(1,4000,100)
#period = np.random.randint(1,10,100)
#risk = np.random.randint(1,4,100)
#mf = np.random.randint(1,5,100)
import pandas as pd
#df = pd.DataFrame()
#df['age']=age
#df['amnt']=amnt
#df['period']=period
#df['risk']=risk
#df['mf']=mf
df = pd.read_csv('C:/Users/HP/mfdata.csv')
#df.to_csv('mfdata.csv')
#print(df)

from sklearn.model_selection import train_test_split
x = df[['age','amnt','period','risk']]
y = df['mf']

#print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape )
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from sklearn.neural_network import multilayer_perceptron
clf1 = LogisticRegression()
clf2 = KNeighborsClassifier()
clf3 = RandomForestClassifier()
clf4 = SVC()

clfs = [clf1,clf2,clf3,clf4]
for clf in clfs:
    clf.fit(x_train,y_train)
    prds = clf.predict(x_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test,prds)
    print('clf: ',acc)
    

user = [[25,2000,2,2]]
print(user)

mff = clf.predict(user)
print(mff)


# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 22:37:02 2020
@author: YugaShree1
"""

from flask import Flask,jsonify
app = Flask(__name__)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def lRandomForestFunction():  
    dataset = pd.read_csv("smsdataset.csv") 
    X = dataset.iloc[:, 0:1].values  
    y = dataset.iloc[:, 1].values 
    percent=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
    l = [None] * 10
    i=0
    thisdict =	{}
    for x in percent:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=x, random_state=0) 
        sc = StandardScaler()  
        X_train = sc.fit_transform(X_train)  
        X_test = sc.transform(X_test) 
        regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
        regressor.fit(X_train, y_train)  
        y_pred = regressor.predict(X_test) 
        l[i]=accuracy_score(y_test, y_pred.round())*100;
        thisdict[str((int(x*100)))] = l[i]
        print("x=",str((int(x*100))))
        i=i+1
        print("Accuracy=",accuracy_score(y_test, y_pred.round())) 
    label = ['0.50%','0.55%','0.60%','0.65%','0.70%','0.75%','0.80%','0.85%','0.90%','0.95%']
    index = np.arange(len(label))
    print(l)
    return thisdict;

@app.route('/randomforest/percentsplit')
def index():
    #return 'Welcome First REST API Response!!!'
    return jsonify(lRandomForestFunction())


if __name__ == "__main__":
    app.run(debug=False)
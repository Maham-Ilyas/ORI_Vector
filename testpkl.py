import os
import joblib
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from flask import Flask, render_template, request
from numpy import loadtxt
from sklearn.externals import joblib
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from wtforms import Form, TextAreaField, validators
import pandas as pd

data_frame = pd.read_csv("FVs.csv")
data_frame.shape
data_frame.head(5)
#from sklearn.cross_validation import train_test_split

#feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['Positive','Nagetive']

# Separating out the target
y = data_frame.loc[:,['positive']].values

df2 = data_frame.drop(['positive'], axis=1)
df2.reset_index(inplace=True)
# Separating out the features
df2 = df2.iloc[: , 1:]
x = df2.values

# print("after remove")
# df2 = df2.iloc[: , 1:]
# print(df2.values)

# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# generate two class dataset
#X, y = make_classification(n_samples=1000, n_classes=2, n_features=20, random_state=27)

# split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.2, random_state=27)

from xgboost import XGBClassifier



#Applying PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 100)
#
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)


# XGBClassifier
# model1 =MLPClassifier (random_state=42)

model1 = XGBClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.28, max_depth=10)
model1.fit(X_train, y_train)
xgb_predict_test = model1.predict(X_test)

#get accuracy
xgb_accuracy_testdata = metrics.accuracy_score(y_test, xgb_predict_test)
#print accuracy

print ("Accuracy of xgboost: {0:.4f}".format(xgb_accuracy_testdata))
pickle.dump(model1, open("model.pkl", "wb"))

print (metrics.classification_report(y_test, xgb_predict_test))



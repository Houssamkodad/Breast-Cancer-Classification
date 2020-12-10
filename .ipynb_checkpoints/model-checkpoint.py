# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split


raw_data = pd.read_csv('data.csv')
dataset = raw_data.copy()
dataset['diagnosis']=dataset['diagnosis'].map({'B':0,'M':1})
raw_x = dataset.iloc[:, 2:31].values
raw_y = dataset.iloc[:, 1].values

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state = 0)
#Fitting model with trainig data
log.fit(X_train, Y_train)

# Saving model to disk
pickle.dump(log, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[10.52,16.75	,83.34	,509,	0.19524	,0.06473	,0.08036,	0.08278	,0.102	,0.09907,	0.0249,	0.9591	,2.183	,23.47,	0.008328,	0.008722,	0.02349	,0.00867,	0.06218	,0.005386	,18.84	,32.47,	88.81,	506.2	,0.1249	,0.0872,	0.09076	,0.06316	,0.3906 ]]))


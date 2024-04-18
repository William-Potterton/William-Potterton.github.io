# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 08:57:00 2023

@author: william
"""

# MT412_Python Assignment


# Ex1

birthplVal = input('Where were you born?')
print('This person was born in ' + birthplVal)


# Ex2

tmpLst = []
n = 10
while n > 4:
    tmpLst.append(n)
    n = n - 1

for cpt in range(len(tmpLst)):
    print(tmpLst[cpt])

for elt in tmpLst:
    print(elt)


# Ex3

prodDict = {"ProductA": [10.99, 50], "ProductB": [5.99, 30], 
            "ProductC": [7.49, 25]}
totRev = 0
for keyElt in prodDict.keys():
    totRev += prodDict[keyElt][0] * prodDict[keyElt][1]
    print(keyElt + ' revenue is ' + str(prodDict[keyElt][0] * 
                                        prodDict[keyElt][1]))

print('The total revenue is ' + str(totRev))
import pandas as pd
prodDf = pd.DataFrame(prodDict)
prodDf.set_index([['Cost', 'Quantity']], inplace=True)
prodDf.iloc[1, 2] = 35
prodDf.loc['Quantity', 'ProductB'] = 45


# Ex4

from scipy import stats


def MeanFun(l):
    return(sum(l)/len(l))


def Beta1Fun(la, lb):
    if len(la) == len(lb):
        sumNum = 0
        sumDen = 0
        for i in range(len(la)):
            sumNum += (la[i] - MeanFun(la)) * (lb[i] - MeanFun(lb))
            sumDen += (la[i] - MeanFun(la)) * (la[i] - MeanFun(la))
        return(sumNum / sumDen)


a = [12, 15, 6, 13, 12, 18, 2, 15, 8, 9, 16, 14, 16]
b = [-12, -15, -6, -13, -12, -18, -2, -16, -8, -9, -16, -14, -16]
print(Beta1Fun(a, b))
print(stats.linregress(a, b).slope)


# Ex5

import pandas as pd
inpPath = '/Users/willi/aa Python Coding/'
inpDf = pd.read_csv(inpPath + 'Assignment_Input.csv', delimiter=',', 
                    header=0, index_col=0)
inpDf

beta1Val = Beta1Fun(inpDf['AdvBudg'], inpDf['Sales'])
beta0Val = inpDf['Sales'].mean() - beta1Val * inpDf['AdvBudg'].mean()
yPredVal = beta0Val + beta1Val * 15000

# Ex6

import numpy as np
print(np.arange(105, 195, 5).reshape(3,6))
cLst = [3, 4, 5, 6, 7, 8]
cArr = np.array(cLst)
print(np.mean(cArr))
print(np.median(cArr))
print(np.std(cArr))
print((cArr - np.mean(cArr)) / np.std(cArr))

"""
TEST 2
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:55:33 2023

@author: willi
"""
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1)
inpPath = '/Users/willi/aa Python Coding/'
inpDf = pd.read_csv(inpPath + 'HR_DB.csv', delimiter=',', header=0, 
                    index_col=0)
inpDf

# 2)
inpDf.isna().sum()
inpDf = inpDf.dropna(axis=0, how='any')
inpDf.describe()

# 3) Division is a nominal variable, it should be converted using one-hot 
# encoding to dummy variables
inpDf = pd.get_dummies(inpDf, columns=['division'], prefix='Dpt')
inpDf

# 4) Wage is an ordinal variable, it should be converted using label encoding
# using a dictionary
inpDf['wage'].unique()
wageDict = {'low': 1, 'medium': 2, 'high': 3}
inpDf['wage'] = inpDf['wage'].map(wageDict)

# 5) 
"""
To be considered as normally distributed, the skewness: a measure of the
asymmetry of the probability distribution, must be higher than -2 and lower than 2;
and the kustrosis: a measure of the tailedness/flatness of the probability distribution,
must be higher than -7 and lower than 7.
"""
print(inpDf['recentPromotion'].skew())
print(inpDf['recentPromotion'].kurt())
"""
Both are outside the boundaries, thus we conclude that this variable cannot be considered
as normally distributed.
"""

# 6) 
# To identify outliers in non-normally distributed data, use the interquartile 
# range (IQR) method
# IQR = Q3 - Q1
# lowBd = Q1 - 1.5*IQR
# upBd = Q3 + 1.5*IQR
# IQR = Q3 - Q1, lowBd = Q1 - 1.5*IQR, upBd = Q3 + 1.5*IQR

# 7) We standardise this variable to put it on a common scale with the other variables
inpDf['avgMtlyHrs'] = (inpDf['avgMtlyHrs'] - 
                       inpDf['avgMtlyHrs'].mean()) /inpDf['avgMtlyHrs'].std()

# # GENERATE SUBFILE
# shuffled_inpDf = inpDf.sample(frac=1).reset_index(drop=True)
# shuffled_inpDf = shuffled_inpDf.sample(frac=1, axis=1)
# i = 1
# for eltCol in shuffled_inpDf.columns:
# if eltCol != 'resigned':
# shuffled_inpDf.rename(columns={eltCol: f'v{i}'}, inplace=True)
# i += 1
# # shuffled_inpDf.columns = [f'v{i+1}' for i in range(len(shuffled_inpDf.columns))]
# shuffled_inpDf.iloc[:5000].to_csv(inpPath + 'Sub_HR_DB.csv', sep=',', index=False)

# 8) 
# The main difference between supervised and unsupervised learning is that
# there are labels in supervised learning.
yDf = inpDf['resigned']
xDf = inpDf.drop(columns='resigned')

# 9) 
# The label: 'resigned' variable is a discrete variable thus the type of
# supervised learning chosen here is a classification

from sklearn.model_selection import train_test_split
# 10) 
# Splitting between train and test sets helps assess how well a model 
# generalizes to new, unseen data.
# Generally, a majority of the data is allocated for training the model.
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

# 11) 
#SVM is a supervised machine learning algorithm used for classification and regression tasks.
# SVM aims to find a hyperplane that best separates different classes in the feature space, 
# maximizing the margin between them.
# SVM can handle linear and non-linear relationships through the use of different kernels.
# SVM is known for its effectiveness in high-dimensional spaces and its ability to 
# handle complex datasets.
# Step 1: Training the model on the train set
# Step 2: Obtaining the predicted values from the test set using the trained model
# Step 3: Error/Accuracy analysis: Comparing the actual labels from the test set with 
# the labels predicted by the model
from sklearn import svm
from sklearn.metrics import accuracy_score
# Set the algorithm parameters
clf = svm.SVC(C=100, kernel='rbf', gamma=1, random_state=0)
# Fit the data
clf.fit(X_train, y_train)
# Analyse the output's score
print(clf.score(X_test, y_test))
# or
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 12)
# A Random Forest, that can be used for classification and regression, is an ensemble 
# learning method in machine learning
# that operates by constructing a multitude of decision trees during training and outputs
# the majority of votes (classification) or mean prediction of the trees (regression).
# And a subset of features is tested at each node
# Random Forests is a powerful and versatile algorithm known for high accuracy and 
# resistance to overfitting.
from sklearn.ensemble import RandomForestClassifier
# Set the algorithm parameters
for nTree in [2, 10, 50]:
    clf = RandomForestClassifier(n_estimators=nTree, random_state=0)
    # Fit the data
    clf.fit(X_train, y_train)
    # Analyse the output's score
    print(nTree)
    print(clf.score(X_test, y_test))
    print('')

clf = RandomForestClassifier(n_estimators=50, random_state=0)
# Fit the data
clf.fit(X_train, y_train)

# 13) 
# Feature importance in a Random Forest refers to a measure of the contribution 
# of each feature (or variable)
# to the predictive performance of the model.
# It quantifies the extent to which a feature helps the model in making accurate predictions.
feature_importances = clf.feature_importances_
# Create a DataFrame to associate feature names with their importance scores
featImpDf = pd.DataFrame({'Feature': xDf.columns, 'Importance': feature_importances})
# Sort features by importance in descending order
featImpDf_Sort = featImpDf.sort_values(by='Importance', ascending=False)
# Identify the top three most important features
print(featImpDf_Sort.head(3))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 21:11:45 2021

@author: carolyndavis
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from pydataset import data



import acquire as a
import prepare


#Acquire the data

tit_df = a.get_titanic_data()

tit_df.head()


                    #PREPARE THE DATA 

#set the index to passenger_id
#drop any useless columns/remove any nulls
#

tit_df = tit_df.set_index('passenger_id')  #resets index 

tit_df = tit_df.drop(columns=['class', 'embarked'])

#Check on the nulls in the data:
    
tit_df.isna().sum()    #how many nulls are there?

# #output:
#     survived         0
# pclass           0
# sex              0
# age            177
# sibsp            0
# parch            0
# fare             0
# deck           688
# embark_town      2
# alone            0
# dtype: int64
#deck has the largest amount of nulls

#drop deck column

tit_df = tit_df.drop(columns=['deck'])

#embarktown should have the nulls filled with the most common value seen in the df

tit_df.embark_town = tit_df.embark_town.fillna(value=tit_df.embark_town.mode())


tit_df.age = tit_df.age.fillna(value=tit_df.age.median())

tit_dummy = pd.get_dummies(tit_df[['sex', 'embark_town']], dummy_na=False, drop_first=[True, True])

#dropping the original columns that were encoded:
tit_df = tit_df.drop(columns=['sex', 'embark_town'])

#connect the dummy df with the og df:

new_df = pd.concat([tit_df, tit_dummy], axis=1)

#######################  SPLIT THE DATA ####################################
train, test = train_test_split(new_df, test_size=.2, random_state=123, stratify=new_df.survived)
train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)

X_train = train.drop(columns=['survived'])
y_train = train.survived

X_validate = validate.drop(columns=['survived'])
y_validate = validate.survived

X_test = test.drop(columns=['survived'])
y_test = test.survived

# =============================================================================
# 1.)Fit a K-Nearest Neighbors classifier to your training sample and transform
#  (i.e. make predictions on the training sample)
# =============================================================================


#                   TRAIN THE MODEL//CREATE KNN OBJECT

X_train.columns
#9
# weights = ['uniform', 'distance']
knn = KNeighborsClassifier(n_neighbors=9, weights='uniform')

#Fit the model on the training data

knn.fit(X_train, y_train)


#Make Predictions:
# predict on X_train
y_pred = knn.predict(X_train)
y_pred


# calculate probabilities (if you need them)
y_pred_proba = knn.predict_proba(X_train)




# =============================================================================
# 2.)Evaluate your results using the model score, confusion matrix, and classification report.
# =============================================================================
#Compute the accuracy

print('Accuracy of KNN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
# output:Accuracy of KNN classifier on training set: 0.77


# Create a confusion matrix
print(confusion_matrix(y_train, y_pred))
# [[262  45]
#  [ 72 119]]

# Create a classification report

pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
#output:                     0           1  accuracy   macro avg  weighted avg
# precision    0.784431    0.725610   0.76506    0.755020      0.761871
# recall       0.853420    0.623037   0.76506    0.738228      0.765060
# f1-score     0.817473    0.670423   0.76506    0.743948      0.761074
# support    307.000000  191.000000   0.76506  498.000000    498.000000



# =============================================================================
# 3.)Print and clearly label the following: Accuracy, true positive rate, false positive
#     rate, true negative rate, false negative rate, precision, recall, f1-score, and
#     support.
# =============================================================================
#Validate the Model 
# predict on X_validate 
y_pred = knn.predict(X_validate)
y_pred


print('Accuracy of KNN classifier on validate set: {:.2f}'
     .format(knn.score(X_validate, y_validate)))
#Accuracy of KNN classifier on validate set: 0.70

                    
# =============================================================================
#                             What is the best value of k?
# =============================================================================

# =============================================================================
# 4.) Run through steps 2-4 setting k to 10
# =============================================================================


metrics = []

# loop through different values of k
for k in range(1, 10):
            
    # define the thing
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # fit the thing (remmeber only fit on training data)
    knn.fit(X_train, y_train)
    
    # use the thing (calculate accuracy)
    train_accuracy = knn.score(X_train, y_train)
    validate_accuracy = knn.score(X_validate, y_validate)
    
    output = {
        "k": k,
        "train_accuracy": train_accuracy,
        "validate_accuracy": validate_accuracy
    }
    
    metrics.append(output)

# make a dataframe
results = pd.DataFrame(metrics)

# plot the data
results.set_index('k').plot(figsize = (16,9))
plt.ylim(0.90, 1)
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,10,1))
plt.grid()



print('Accuracy of KNN classifier on test set: {:.2f}'
     .format(knn.score(X_validate, y_validate)))
# =============================================================================
# 5.)Run through setps 2-4 setting k to 20
# =============================================================================

metrics = []

# loop through different values of k
for k in range(1, 20):
            
    # define the thing
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # fit the thing (remmeber only fit on training data)
    knn.fit(X_train, y_train)
    
    # use the thing (calculate accuracy)
    train_accuracy = knn.score(X_train, y_train)
    validate_accuracy = knn.score(X_validate, y_validate)
    
    output = {
        "k": k,
        "train_accuracy": train_accuracy,
        "validate_accuracy": validate_accuracy
    }
    
    metrics.append(output)

# make a dataframe
results = pd.DataFrame(metrics)

# plot the data
results.set_index('k').plot(figsize = (16,9))
plt.ylim(0.90, 1)
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,20,1))
plt.grid()

print('Accuracy of KNN classifier on test set: {:.2f}'
     .format(knn.score(X_validate, y_validate)))
# =============================================================================
# 6.)What are the differences in the evaluation metrics? Which performs better on your in-sample
#  data? Why?
# =============================================================================
# knn_10 accuracy : Accuracy of KNN classifier on test set: 0.70

#knn_20 accucracy: Accuracy of KNN classifier on test set: 0.73

#larger the depth the greater the accuracy/20 performs better


# =============================================================================
# # 7.)Which model performs best on our out-of-sample data from validate?
# =============================================================================
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
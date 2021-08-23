#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:35:43 2021

@author: carolyndavis
"""

# =============================================================================
#                 DECISION TREE EXERCISES 
# =============================================================================
# =============================================================================
# 1.)What is your baseline prediction? What is your baseline accuracy? remember: your
#  baseline prediction for a classification problem is predicting the most prevelant
#  class in the training dataset (the mode). When you make those predictions, what is
#  your accuracy? This is your baseline accuracy.
# =============================================================================


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import graphviz
from graphviz import Graph

import warnings
warnings.filterwarnings("ignore")

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

# =============================================================================
# 
# 2.)Fit the decision tree classifier to your training sample and transform
#  (i.e. make predictions on the training sample)
# =============================================================================
#Examining the age columns/ 177 nulls, causation

age_naan = tit_df[tit_df.age.isna()]
age_naan.alone.value_counts()

#output:
# #1    133
# 0     44
# Name: alone, dtype: int64
#Look at the age_naan data:

tit_df.fare.hist(), age_naan.fare.hist()

#compare this group to the rest of the population....

for column in tit_df.drop(columns=['age', 'fare']).columns:
    print(column)
    print('Population:')
    print(tit_df[column].value_counts(normalize=True))
    print("No age:")
    print(age_naan[column].value_counts(normalize=True))
    print("-" * 20)
    print()
    
#Nothing wildly different in the age col compared to the entire population 
#use the median age maybe...

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

x_validate = validate.drop(columns=['survived'])
y_validate = validate.survived

X_test = test.drop(columns=['survived'])
y_test = test.survived
# =============================================================================
# 3.)Evaluate your in-sample results using the model score, confusion matrix, and
#  classification report.
# =============================================================================

                #Now model the data
#most common value is the great for establlishing a baseline, most common value

baseline = y_train.mode()

match_bsl_prediction = y_train == 0
#^^^^ gives a array of bools with True return for values matching baseline accuracy

baseline_accuracy = match_bsl_prediction.mean()

#baseline_accuracy = 0.6164658634538153// .62

                #Baseline ACC established: make the models...
                
tree1 = DecisionTreeClassifier(max_depth=1, random_state=123)

#Fitting the model on train and only just train
tree1 = tree1.fit(X_train, y_train)


#Now evaluate the model to see how it performs on just TRAIN
y_predictions = tree1.predict(X_train)

#Make the classification report for actual y-values and the models predicted y-values:

class_report = classification_report(y_train, y_predictions, output_dict=(True))
print("Tree1 depth")
pd.DataFrame(class_report)
                            #Tree1 depth

#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000


for i in range(2, 11):
    tree =  DecisionTreeClassifier(max_depth=3, random_state=123)
    tree1.fit(X_train, y_train)
    y_predictions = tree1.predict(X_train)
    class_report = classification_report(y_train, y_predictions, output_dict=(True))
    print(f"Tree with max depth of {i}")
    print(pd.DataFrame(class_report))
    print()

#output for tree with max_depth of 1/// Train:
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000 
    
#tree with max_depth of 2:
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000
    

#tree with max_depth of 3:
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000

#tree with max_depth of 4:
#      0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000

#tree with max_depth of 5:
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000

#tree with max_depth of 6:
#            0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000
    

#tree with max_depth of 7:
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000

#tree with max_depth of 8:
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000    
    

#tree with max_depth of 9:
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000


#tree with max_depth of 10:
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.820433    0.760000  0.799197    0.790217      0.797255
# recall       0.863192    0.696335  0.799197    0.779764      0.799197
# f1-score     0.841270    0.726776  0.799197    0.784023      0.797358
# support    307.000000  191.000000  0.799197  498.000000    498.000000

#observations:
    #the more depth afforded to the decision tree, the closer the model fits to the 
    #to the training data
    #What woukd the model do with data that it has not seen previously

# =============================================================================
# 4.)Compute: Accuracy, true positive rate, false positive rate, true negative rate,
#  false negative rate, precision, recall, f1-score, and support.
# =============================================================================

metrics = []  #collector for appended values from loop iteration

for i in range(2,25):
    tree = DecisionTreeClassifier(max_depth=i, random_state=123)
    
    #run the model on train and only TRAIN data 
    tree = tree.fit(X_train, y_train)
    
    #use/test the model to evaluate models performance on train data first...
    in_sample_accuracy = tree.score(X_train, y_train)
    out_sample_accuracy = tree.score(x_validate, y_validate)
    
    output = {'max_depth': i, 'train_accuracy': in_sample_accuracy, 'validate_accuracy': out_sample_accuracy}
    
    metrics.append(output)
    
tree_df = pd.DataFrame(metrics)
tree_df["difference"] = tree_df.train_accuracy - tree_df.validate_accuracy

tree_df.head(2)
#output:
#    max_depth  train_accuracy  validate_accuracy  difference
# 0          2        0.799197           0.761682    0.037515
# 1          3        0.825301           0.799065    0.026236


#Code ca =n be continuously modified by setting a threshold of differences 
#Continue to lopp to compare the in-sample and out of sample data

# =============================================================================
# 
# 5.)Run through steps 2-4 using a different max_depth value.
# =============================================================================

threshold = 0.10  #threshold set for amount of overfit that is tolerated

models = []
metrics = []

for i in range(2, 25):
    tree = DecisionTreeClassifier(max_depth=i, random_state=123)
    #^^^ creates the model
    
    tree = tree.fit(X_train, y_train)   #fit model to train data and only TRAIN data
    
    in_sample_accuracy = tree.score(X_train, y_train)
    out_sample_accuracy = tree.score(x_validate, y_validate)
    #^^^evaluates the models performance on train data first
    
    difference = in_sample_accuracy - out_sample_accuracy
    #^^calculates the difference in accuracy
    
    if difference > threshold:
        break
    #^^adds conditions to check the accuracy vs the threshold
    
    output = {
        'max_depth': i,
        'train_accuracy': in_sample_accuracy,
        'validate_accuracy': out_sample_accuracy,
        'difference': difference}
    #^^^formats the output for each models performance o train and validate
    
    metrics.append(tree)
    
    models.append(tree)
    
model_df = pd.DataFrame(metrics)


model_df.head()


# =============================================================================
# Which model performs better on your in-sample data?
# =============================================================================






# =============================================================================
# Which model performs best on your out-of-sample data, the validate set?
# =============================================================================

# 0  DecisionTreeClassifier(max_depth=2, random_state=123)
# 1  DecisionTreeClassifier(max_depth=3, random_state=123)
# 2  DecisionTreeClassifier(max_depth=4, random_state=123)
# 3  DecisionTreeClassifier(max_depth=5, random_state=123)
# 4  DecisionTreeClassifier(max_depth=6, random_state=123)
#lowest dfference in accuracy = model 2
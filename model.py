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




# =============================================================================
#                     RANDOM FOREST CLASSIFIER EXERCISES
# =============================================================================
:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import graphviz
from graphviz import Graph

import warnings
warnings.filterwarnings("ignore")

import acquire as a
import prepare
# =============================================================================
# 1.)Fit the Random Forest classifier to your training sample and transform (i.e. make
#   predictions on the training sample) setting the random_state accordingly and setting
#  min_samples_leaf = 1 and max_depth = 10.
# =============================================================================

#ACQUIRE THE DATA

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
#                 #Now EStablish a Baseline for comparison 
# =============================================================================
#most common value is the great for establlishing a baseline, most common value

baseline = y_train.mode()

match_bsl_prediction = y_train == 0
#^^^^ gives a array of bools with True return for values matching baseline accuracy

baseline_accuracy = match_bsl_prediction.mean()

#baseline_accuracy = 0.6164658634538153// .62

                #Baseline ACC established: make the models...
# =============================================================================
# 2.)Evaluate your results using the model score, confusion matrix, and classification
#  report.
# =============================================================================
                #Now make that Random Forest Model:
forest1 = RandomForestClassifier(max_depth=1, random_state=123)  #makes the model

tree1 = forest1.fit(X_train, y_train)
#fit the forest model on the train and only train data

#using/evaluating the model on the train data first
y_predictions = forest1.predict(X_train)

#make a classification report for the actual y values awa model's predicted y-values
report = classification_report(y_train, y_predictions, output_dict=True)
print("Tree with max_depth of 1")
pd.DataFrame(report)

# =============================================================================
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.773481    0.801471  0.781124    0.787476      0.784216
# recall       0.912052    0.570681  0.781124    0.741366      0.781124
# f1-score     0.837070    0.666667  0.781124    0.751868      0.771715
# support    307.000000  191.000000  0.781124  498.000000    498.000000
# =============================================================================
for i in range(2, 11):
    forest =  RandomForestClassifier(max_depth=i, random_state=123)
    forest = forest.fit(X_train, y_train)
    y_predictions = forest.predict(X_train)
    report = classification_report(y_train, y_predictions, output_dict=(True))
    print(f"Tree with max depth of {i}")
    print(pd.DataFrame(report))
    print()




# =============================================================================
# 3.)Print and clearly label the following: Accuracy, true positive rate, false positive
#     rate, true negative rate, false negative rate, precision, recall, f1-score, and
#     support.
# =============================================================================
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.773481    0.801471  0.781124    0.787476      0.784216
# recall       0.912052    0.570681  0.781124    0.741366      0.781124
# f1-score     0.837070    0.666667  0.781124    0.751868      0.771715
# support    307.000000  191.000000  0.781124  498.000000    498.000000

for i in range(2, 11):
    forest =  RandomForestClassifier(max_depth=i, random_state=123)
    forest = forest.fit(X_train, y_train)
    y_predictions = forest.predict(X_train)
    report = classification_report(y_train, y_predictions, output_dict=(True))
    print(f"Tree with max depth of {i}")
    print(pd.DataFrame(report))
    print()
# Tree with max depth of 2
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.815029    0.835526  0.821285    0.825278      0.822890
# recall       0.918567    0.664921  0.821285    0.791744      0.821285
# f1-score     0.863706    0.740525  0.821285    0.802115      0.816462
# support    307.000000  191.000000  0.821285  498.000000    498.000000

# Tree with max depth of 3
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.836257    0.865385  0.845382    0.850821      0.847429
# recall       0.931596    0.706806  0.845382    0.819201      0.845382
# f1-score     0.881356    0.778098  0.845382    0.829727      0.841753
# support    307.000000  191.000000  0.845382  498.000000    498.000000

# Tree with max depth of 4
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.842566    0.883871  0.855422    0.863218      0.858408
# recall       0.941368    0.717277  0.855422    0.829323      0.855422
# f1-score     0.889231    0.791908  0.855422    0.840569      0.851904
# support    307.000000  191.000000  0.855422  498.000000    498.000000

# Tree with max depth of 5
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.846821    0.907895  0.865462    0.877358      0.870245
# recall       0.954397    0.722513  0.865462    0.838455      0.865462
# f1-score     0.897397    0.804665  0.865462    0.851031      0.861831
# support    307.000000  191.000000  0.865462  498.000000    498.000000

# Tree with max depth of 6
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.866667    0.947712  0.891566    0.907190      0.897750
# recall       0.973941    0.759162  0.891566    0.866552      0.891566
# f1-score     0.917178    0.843023  0.891566    0.880101      0.888737
# support    307.000000  191.000000  0.891566  498.000000    498.000000

# Tree with max depth of 7
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.906907    0.969697  0.927711    0.938302      0.930989
# recall       0.983713    0.837696  0.927711    0.910705      0.927711
# f1-score     0.943750    0.898876  0.927711    0.921313      0.926539
# support    307.000000  191.000000  0.927711  498.000000    498.000000

# Tree with max depth of 8
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.926829    0.982353  0.945783    0.954591      0.948124
# recall       0.990228    0.874346  0.945783    0.932287      0.945783
# f1-score     0.957480    0.925208  0.945783    0.941344      0.945103
# support    307.000000  191.000000  0.945783  498.000000    498.000000

# Tree with max depth of 9
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.938650    0.994186  0.957831    0.966418      0.959950
# recall       0.996743    0.895288  0.957831    0.946015      0.957831
# f1-score     0.966825    0.942149  0.957831    0.954487      0.957361
# support    307.000000  191.000000  0.957831  498.000000    498.000000

# Tree with max depth of 10
#                     0           1  accuracy   macro avg  weighted avg
# precision    0.953416    1.000000   0.96988    0.976708      0.971283
# recall       1.000000    0.921466   0.96988    0.960733      0.969880
# f1-score     0.976153    0.959128   0.96988    0.967640      0.969623
# support    307.000000  191.000000   0.96988  498.000000    498.000000

#observations:
    #the more depth afforded to the decision tree, the closer the model fits to the 
    #to the training data
    #What woukd the model do with data that it has not seen previously



#More loops to compare in sample to out of sample

metrics = []

for i in range(2, 25):
    forest = RandomForestClassifier(max_depth=i, random_state=123) #<<make the model
    
    forest = forest.fit(X_train, y_train)
    #^^fitting the model on train and train only 
    
    in_sample_accuracy = forest.score(X_train, y_train)
    #^use model/evaluating models performance 
    
    out_of_sample_accuracy = forest.score(x_validate, y_validate)
    
    output = {
        "max_depth": i,
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy}
    metrics.append(output)
    
df = pd.DataFrame(metrics)
df['difference'] = df.train_accuracy - df.validate_accuracy


#Setting the threshold of difference for the code above 
#continue to compare in and out of samopke 

threshold = 0.10 #Set threshold for the overfit that will be tolerated

models = []
metrics = []

for i in range(2, 25):
    
    forest = RandomForestClassifier(max_depth=i, min_samples_leaf=1, random_state=123)
    #^^make the model
    # Fit the model (on train and only train)
    forest = forest.fit(X_train, y_train)

    # Use the model
    # We'll evaluate the model's performance on train, first
    in_sample_accuracy = forest.score(X_train, y_train)   
    out_of_sample_accuracy = forest.score(x_validate, y_validate)

    # Calculate the difference
    difference = in_sample_accuracy - out_of_sample_accuracy
    
    # Add a conditional to check vs. the threshold
    if difference > threshold:
        break
    
    # Formulate the output for each model's performance on train and validate
    output = {
        "max_depth": i,
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy,
        "difference": difference
    }
    
    
#Add the metrics dict to the lst, so df can be created:
    metrics.append(output)
    
    models.append(forest)
    #adding the specific tree to a list of trained models
df = pd.DataFrame(metrics)


# =============================================================================
# 4.)Run through steps increasing your min_samples_leaf and decreasing your max_depth.
# =============================================================================


# =============================================================================
#             INCREASING MIN_SAMPLES_LEAF/DECREASE MAX_DEPTH
# =============================================================================

# compare in-sample to out-of-sample
metrics = []
max_depth = 20

for i in range(2, max_depth):
    # Make the model
    depth = max_depth - i
    n_samples = i
    forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=n_samples, random_state=123)

    # Fit the model (on train and only train)
    forest = forest.fit(X_train, y_train)

    # Use the model
    # We'll evaluate the model's performance on train, first
    in_sample_accuracy = forest.score(X_train, y_train)
    
    out_of_sample_accuracy = forest.score(x_validate, y_validate)

    output = {
        "min_samples_per_leaf": n_samples,
        "max_depth": depth,
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy
    }
    
    metrics.append(output)
    
df = pd.DataFrame(metrics)
df["difference"] = df.train_accuracy - df.validate_accuracy
df

#output:
#     min_samples_per_leaf  max_depth  ...  validate_accuracy  difference
# 0                      2         18  ...           0.822430    0.103273
# 1                      3         17  ...           0.817757    0.083849
# 2                      4         16  ...           0.817757    0.069793
# 3                      5         15  ...           0.780374    0.097136
# 4                      6         14  ...           0.799065    0.072421
# 5                      7         13  ...           0.789720    0.079758
# 6                      8         12  ...           0.794393    0.071069
# 7                      9         11  ...           0.794393    0.063037
# 8                     10         10  ...           0.785047    0.072383
# 9                     11          9  ...           0.785047    0.064351
# 10                    12          8  ...           0.780374    0.058984
# 11                    13          7  ...           0.780374    0.058984
# 12                    14          6  ...           0.785047    0.048287
# 13                    15          5  ...           0.789720    0.041606
# 14                    16          4  ...           0.789720    0.041606
# 15                    17          3  ...           0.780374    0.046935
# 16                    18          2  ...           0.766355    0.038866
# 17                    19          1  ...           0.761682    0.019442

# [18 rows x 5 columns]






sns.scatterplot(x="max_depth", y="difference", data=df)


sns.scatterplot(x="min_samples_per_leaf", y="difference", data=df)



sns.scatterplot(x="difference", y="validate_accuracy", data=df)

# =============================================================================
#                 INCREASE BOTH MIN_SAMP_PER_LEAF // MAX_DEPTH
# =============================================================================
metrics = []
max_depth = 20

for i in range(2, max_depth):
    # Make the model
    depth = i
    n_samples = i
    forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=n_samples, random_state=123)

    # Fit the model (on train and only train)
    forest = forest.fit(X_train, y_train)

    # Use the model
    # We'll evaluate the model's performance on train, first
    in_sample_accuracy = forest.score(X_train, y_train)
    
    out_of_sample_accuracy = forest.score(x_validate, y_validate)

    output = {
        "min_samples_per_leaf": n_samples,
        "max_depth": depth,
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy
    }
    
    metrics.append(output)
    
df = pd.DataFrame(metrics)
df["difference"] = df.train_accuracy - df.validate_accuracy
df

#output:
#     min_samples_per_leaf  max_depth  ...  validate_accuracy  difference
# 0                      2          2  ...           0.771028    0.050257
# 1                      3          3  ...           0.785047    0.060335
# 2                      4          4  ...           0.794393    0.052997
# 3                      5          5  ...           0.799065    0.060372
# 4                      6          6  ...           0.799065    0.062380
# 5                      7          7  ...           0.789720    0.073734
# 6                      8          8  ...           0.789720    0.073734
# 7                      9          9  ...           0.794393    0.061029
# 8                     10         10  ...           0.785047    0.072383
# 9                     11         11  ...           0.785047    0.064351
# 10                    12         12  ...           0.780374    0.058984
# 11                    13         13  ...           0.780374    0.058984
# 12                    14         14  ...           0.780374    0.054968
# 13                    15         15  ...           0.789720    0.045622
# 14                    16         16  ...           0.785047    0.050295
# 15                    17         17  ...           0.780374    0.044927
# 16                    18         18  ...           0.789720    0.033574
# 17                    19         19  ...           0.775701    0.059640

# [18 rows x 5 columns]

# =============================================================================
#             FIXED DEPTH //INCREASING MIN_SAMP_LEAF
# =============================================================================
metrics = []
max_depth = 50

for i in range(2, max_depth):
    # Make the model
    depth = 10
    n_samples = i
    forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=n_samples, random_state=123)

    # Fit the model (on train and only train)
    forest = forest.fit(X_train, y_train)

    # Use the model
    # We'll evaluate the model's performance on train, first
    in_sample_accuracy = forest.score(X_train, y_train)
    
    out_of_sample_accuracy = forest.score(x_validate, y_validate)

    output = {
        "min_samples_per_leaf": n_samples,
        "max_depth": depth,
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy
    }
    
    metrics.append(output)
    
df = pd.DataFrame(metrics)
df["difference"] = df.train_accuracy - df.validate_accuracy
df

#     min_samples_per_leaf  max_depth  ...  validate_accuracy  difference
# 0                      2         10  ...           0.822430    0.099257
# 1                      3         10  ...           0.808411    0.091187
# 2                      4         10  ...           0.813084    0.074466
# 3                      5         10  ...           0.794393    0.083118
# 4                      6         10  ...           0.799065    0.072421
# 5                      7         10  ...           0.789720    0.079758
# 6                      8         10  ...           0.794393    0.073077
# 7                      9         10  ...           0.794393    0.063037
# 8                     10         10  ...           0.785047    0.072383
# 9                     11         10  ...           0.785047    0.064351
# 10                    12         10  ...           0.780374    0.058984
# 11                    13         10  ...           0.780374    0.058984
# 12                    14         10  ...           0.780374    0.054968
# 13                    15         10  ...           0.789720    0.045622
# 14                    16         10  ...           0.785047    0.050295
# 15                    17         10  ...           0.780374    0.044927
# 16                    18         10  ...           0.789720    0.033574
# 17                    19         10  ...           0.775701    0.059640
# 18                    20         10  ...           0.761682    0.071651
# 19                    21         10  ...           0.785047    0.030214
# 20                    22         10  ...           0.780374    0.038903
# 21                    23         10  ...           0.766355    0.054930
# 22                    24         10  ...           0.766355    0.052922
# 23                    25         10  ...           0.766355    0.040874
# 24                    26         10  ...           0.761682    0.045547
# 25                    27         10  ...           0.761682    0.043539
# 26                    28         10  ...           0.766355    0.040874
# 27                    29         10  ...           0.761682    0.045547
# 28                    30         10  ...           0.757009    0.056244
# 29                    31         10  ...           0.757009    0.054236
# 30                    32         10  ...           0.766355    0.038866
# 31                    33         10  ...           0.761682    0.037515
# 32                    34         10  ...           0.757009    0.050220
# 33                    35         10  ...           0.757009    0.050220
# 34                    36         10  ...           0.757009    0.050220
# 35                    37         10  ...           0.766355    0.040874
# 36                    38         10  ...           0.766355    0.042882
# 37                    39         10  ...           0.766355    0.042882
# 38                    40         10  ...           0.766355    0.040874
# 39                    41         10  ...           0.766355    0.040874
# 40                    42         10  ...           0.766355    0.034850
# 41                    43         10  ...           0.761682    0.043539
# 42                    44         10  ...           0.757009    0.042187
# 43                    45         10  ...           0.766355    0.034850
# 44                    46         10  ...           0.757009    0.026123
# 45                    47         10  ...           0.757009    0.026123
# 46                    48         10  ...           0.757009    0.024115
# 47                    49         10  ...           0.757009    0.024115

# [48 rows x 5 columns]




# =============================================================================
# 5.)What are the differences in the evaluation metrics? Which performs better on your
#     in-sample data? Why?
# =============================================================================





# =============================================================================
# 
# After making a few models, which one has the best performance (or closest metrics)
#  on both train and validate?
# =============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 09:43:41 2021

@author: carolyndavis
"""

# =============================================================================
# 1.)Create a new file named model_evaluation.py or model_evaluation.ipynb
# for these exercises.
# =============================================================================

import pandas as pd 
from sklearn.metrics import confusion_matrix


# =============================================================================
# 2.)Given the following confusion matrix, evaluate (by hand) the model's performance.
# =============================================================================


# In the context of this problem, what is a false positive?


# In the context of this problem, what is a false negative?

# How would you describe this model?


#consider cat the positive class
#consder dog is the negative class

#FP: Predicted cat, but ACTUALLY DOG
#FN: Predicted dog, but ACTUALLY CAT

true_pos = 34  #bc cat is in the positive class
true_neg = 46  #bc dog is neg class
false_pos = 7   #we get 7 cats but is actually dog
false_neg = 13 #we get cats and it is not cats


accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
recall = true_pos / (true_pos + false_neg)
precision = true_pos / (true_pos + false_pos)

print("Accuracy = ", accuracy)
print("Recall = ", recall)
print("Precision = ", precision)

# Accuracy =  0.8
# Recall =  0.723404255319149
# Precision =  0.8292682926829268



# =============================================================================
# 3.) You are working as a datascientist working for Codeup Cody Creator (C3 for short),
#  a rubber-duck manufacturing plant.
# 
# Unfortunately, some of the rubber ducks that are produced will have defects. Your team
#  has built several models that try to predict those defects, and the data from their
#  predictions can be found here.
    #Use the predictions dataset and pandas to help answer the following questions:
# =============================================================================


# An internal team wants to investigate the cause of the manufacturing defects.
#  They tell you that they want to identify as many of the ducks that have a defect
#  as possible. Which evaluation metric would be appropriate here? Which model would
#  be the best fit for this use case?

c3_df = pd.read_csv('c3.csv')    #grab the data
c3_df.head()

c3_df.info()

c3_df.actual.value_counts()   #lets look at the categorical variables 
#No Defect    184
# Defect        16

#our interest is in the defects so:
    
#defects is the positive class
#no dects is the negative class

#We need the metric to identify as many defect ducks as possible
#Recall would be the best is recall--the num of ducks that actually flagged by defect 
#(pos class models)

# =============================================================================
#                             #First model
# =============================================================================
subset = c3_df[c3_df.actual == 'Defect']


                        #model 1 Recall
model_recall = (subset.actual == subset.model1).mean()
#model recall = .5 or 50%

# =============================================================================
#                             #Second Model
# =============================================================================
#model 2 recall

model_recall = (subset.actual == subset.model2).mean()
#model2 recall = .5625 or 56.25%


# =============================================================================
#                             #Third Model
# =============================================================================
#model 3 recall

model_recall = (subset.actual == subset.model3).mean()
#model3 recall = .8125 or 81.25%

#Observations:
    
    #choose model with higher recall to prevent false negatives
    #Model 3 is recommended for this COA
    
# =============================================================================
# Recently several stories in the local news have come out highlighting customers
#  who received a rubber duck with a defect, and portraying C3 in a bad light.
#  The PR team has decided to launch a program that gives customers with a
#  defective duck a vacation to Hawaii. They need you to predict which ducks
#  will have defects, but tell you the really don't want to accidentally give out
#  a vacation package when the duck really doesn't have a defect. Which evaluation
#  metric would be appropriate here? Which model would be the best fit for this use case?
# =============================================================================


#PR wants to minimize FP's, need model with the highest precision..

subset = c3_df[c3_df.model1 == 'Defect']
#^^subset of model where only positive predictions are selected

precision_model = (subset.actual == subset.model1).mean()
#precision = .8 or 80% model 1


                #model 2
subset = c3_df[c3_df.model2 == 'Defect']

precision_model = (subset.actual == subset.model2).mean()
#precision = .1 or 10% model 2


subset = c3_df[c3_df.model3 == 'Defect']

precision_model = (subset.actual == subset.model3).mean()
#precision = 0.13131313131313133 or 13.13%

            #observations:
                #model 1 indicates it will deccrease FP predictions for duck defects



# =============================================================================
# 4.)You are working as a data scientist for Gives You Paws ™, a subscription based
#     service that shows you cute pictures of dogs or cats (or both for an additional
#                                                           fee).
# 
# At Gives You Paws, anyone can upload pictures of their cats or dogs. The photos
#  are then put through a two step process. First an automated algorithm tags pictures
#  as either a cat or a dog (Phase I). Next, the photos that have been initially 
#  identified are put through another round of review, possibly with some human 
#  oversight, before being presented to the users (Phase II).
# =============================================================================

# =============================================================================
# Given this dataset, use pandas to create a baseline model (i.e. a model that just
# predicts the most common class) and answer the following questions:
# =============================================================================
#grab that data
paws_df = pd.read_csv("https://ds.codeup.com/data/gives_you_paws.csv")
paws_df.head()


#Look at the distribution of class/category/values

paws_df.actual.value_counts()
# dog    3254
# cat    1746
# Name: actual, dtype: int64

#most freq label is dog 
# =============================================================================
# a.) In terms of accuracy, how do the various models compare to the baseline model?
#     Are any of the models better than the baseline?
# =============================================================================

            #Create that baseline:
#Baseline model predicts the most common class every time
paws_df['baseline'] = paws_df.actual.value_counts().idxmax()
paws_df.head()
#   actual model1 model2 model3 model4 baseline
# 0    cat    cat    dog    cat    dog      dog
# 1    dog    dog    cat    cat    dog      dog
# 2    dog    cat    cat    cat    dog      dog
# 3    dog    dog    dog    cat    dog      dog
# 4    cat    cat    cat    dog    dog      dog

#in terms of accuracy, how do the various models compare...

#first we need the column names of the df:
    
models = list(paws_df.columns)
models = models[1:]  #Because model cols begin at index [1]/drops actual column
print(models)
# output = ['model1', 'model2', 'model3', 'model4', 'baseline']


                #Determine the ACCURACY in DICT form for each Model
accuracy_out = {} #collector
#can also use .append method
for model in models:
    accuracy = (paws_df.actual == paws_df[model]).mean()
    accuracy_out.update({model:accuracy})
print(accuracy_out)

#output: {'model1': 0.8074, 'model2': 0.6304, 'model3': 0.5096, 'model4': 0.7426, 'baseline': 0.6508}

#Turn it back into a DF with columns for accuracy and model


pd.DataFrame(accuracy_out.items(), columns = ['model', 'accuracy'])
#       model  accuracy
# 0    model1    0.8074
# 1    model2    0.6304
# 2    model3    0.5096
# 3    model4    0.7426
# 4  baseline    0.6508

#Dog = postive class
#cat = negative class

#algo will tag pics as either cat or dog
#for phase 1 choose the model type with the highest recall:
    
#create the subset:
subset = paws_df[paws_df.actual == 'dog']
subset.head()

#subset_output =
#   actual model1 model2 model3 model4 baseline
# 1    dog    dog    cat    cat    dog      dog
# 2    dog    cat    cat    cat    dog      dog
# 3    dog    dog    dog    cat    dog      dog
# 5    dog    dog    dog    dog    dog      dog
# 8    dog    dog    cat    dog    dog      dog

                        # Model 1 Recall:
(subset.actual == subset.model1).mean()
#0.803318992009834

                        #Model 2 Recall:
(subset.actual == subset.model2).mean()
#0.49078057775046097
                        #Model 3 Recall:
(subset.actual == subset.model3).mean()
#0.5086047940995697

                        #Model 4 Recall:
(subset.actual == subset.model4).mean()
#0.9557467732022127    ***MODEL $ RETURNS HIGHEST RECALL

#Now jmake a confusion matrix for model 4 highest recall

pd.crosstab(paws_df.model4, paws_df.actual)
#output=
# actual   cat   dog
# model4            
# cat      603   144
# dog     1143  3110

# =============================================================================
# Next, the photos that have been initially 
# #  identified are put through another round of review, possibly with some human 
# #  oversight, before being presented to the users (Phase II).
# =============================================================================

paws_df.head()
#   actual model1 model2 model3 model4 baseline
# 0    cat    cat    dog    cat    dog      dog
# 1    dog    dog    cat    cat    dog      dog
# 2    dog    cat    cat    cat    dog      dog
# 3    dog    dog    dog    cat    dog      dog
# 4    cat    cat    cat    dog    dog      dog

subset1 = paws_df[paws_df.model1 == 'dog']
subset2 = paws_df[paws_df.model2 == 'dog']
subset3 = paws_df[paws_df.model3 == 'dog']
subset4 = paws_df[paws_df.model4 == 'dog']


# =============================================================================
# b.)Suppose you are working on a team that solely deals with dog pictures.
#     Which of these models would you recomend for Phase I? For Phase II?
# =============================================================================





###LETS looks at the models for Precision:
    
(subset1.actual == subset1.model1).mean()   #Model 1 Precision 
#out: 0.8900238338440586

(subset2.actual == subset2.model2).mean()   #Model 2 Precision
#out:0.8931767337807607

(subset3.actual == subset3.model3).mean()   #Model 3 Precision
#out:0.6598883572567783

(subset4.actual == subset4.model4).mean()   #Model 4 Precision
#out:0.7312485304490948

#confusion matrix 
pd.crosstab(paws_df.model2, paws_df.actual)
#output:
# actual   cat   dog
# model2            
# cat     1555  1657
# dog      191  1597
#Model 2 has the highest precision of the four models tested







# =============================================================================
# c.)Suppose you are working on a team that solely deals with cat pictures.
#     Which of these models would you recomend for Phase I? For Phase II?
# =============================================================================
#cats= positive class

from sklearn.metrics import classification_report

x = classification_report(paws_df.actual, paws_df.model1, labels = ['cat', 'dog'], output_dict=True)

pd.DataFrame(x).T


                #Model 1
pd.DataFrame(classification_report(paws_df.actual, paws_df.model1, labels = ['cat', 'dog'], output_dict=True))



                #Model 2
pd.DataFrame(classification_report(paws_df.actual, paws_df.model2, labels = ['cat', 'dog'], output_dict=True))
#Recall = 0.890607

                #Model 3
pd.DataFrame(classification_report(paws_df.actual, paws_df.model3, labels = ['cat', 'dog'], output_dict=True))
#Recall = 


                #Model 4
pd.DataFrame(classification_report(paws_df.actual, paws_df.model4, labels = ['cat', 'dog'], output_dict=True))
#Precision = 0.807229








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:17:49 2021

@author: carolyndavis
"""

# =============================================================================
# 1.)Create a model that includes age in addition to fare and pclass. 
# Does this model perform better than your baseline?
# =============================================================================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import acquire as a


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
# =============================================================================
# 2.)Include sex in your model as well. Note that you'll need to encode or create a 
# dummy variable of this feature before including it in a model.
# =============================================================================

# =============================================================================
# 3.)Try out other combinations of features and models.
# =============================================================================

# =============================================================================
# 4.)Use you best 3 models to predict and evaluate on your validate sample.
# =============================================================================

# =============================================================================
# 5.)Choose you best model from the validation performation, and evaluate it on the 
# test dataset. How do the performance metrics compare to validate? to train?
# =============================================================================

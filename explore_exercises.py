#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:45:03 2021

@author: carolyndavis
"""

# =============================================================================
# Exploratory ANALYSIS EXERCISES
# =============================================================================



# =============================================================================
# 1.) Acquire, prepare & split your data.
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import acquire as a
#went through preparation steps for practice

iris_data = a.get_iris_data()

iris_data = iris_data.rename(columns={'species_name':'species'})

iris_data = iris_data.drop(columns='species_id')

#Dummy variables for species name
iris_dummy = pd.get_dummies(iris_data['species'], drop_first= False)

#Join the dummy df with the iris df

iris_data = pd.concat([iris_data, iris_dummy], axis=1)

iris_data.plot()

                    #SPLIT THE DATA
                    
# 20% test, 80% train_validate
# then of the 80% train_validate: 30% validate, 70% train. 

train, test = train_test_split(iris_data, test_size=.2, random_state=123, stratify=iris_data.species)
train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.species)                    

def train_validate_test_split(iris_data, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test


train, validate, test = train_validate_test_split(iris_data, target='species')    #species is cateogorical target variable
train.head(2) 
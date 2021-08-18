#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:08:12 2021

@author: carolyndavis
"""

# =============================================================================
# Using the Iris Data:
# =============================================================================




# =============================================================================
# 1.)Use the function defined in acquire.py to load the iris data.
# =============================================================================

import pandas as pd
import numpy as np

import acquire as a

# import data from acquire file as df, also creates a csv of the data if not already present
iris_df = a.get_iris_data() 
iris_df.head()
    
    
# =============================================================================
# 2.)Drop the species_id and measurement_id columns.
# =============================================================================

iris_df = iris_df.drop(columns='species_id') # .drop(columns=column_name)
iris_df.head()    

    
# =============================================================================
# 3.)Rename the species_name column to just species.
# =============================================================================
iris_df = iris_df.rename(columns={'species_name':'species'})
iris_df.head()    
# =============================================================================
# 4.)Create dummy variables of the species name.
# =============================================================================
#.get_dummies(column_name,not dropping any of the dummy columns)

iris_dummy = pd.get_dummies(iris_df['species'], drop_first=False)

iris_df = pd.concate([iris_df, iris_dummy], axis=1) #combines according to iris_df index
    
# =============================================================================
#  5.)Create a function named prep_iris that accepts the untransformed iris data,
#     and returns the data with the transformations above applied.
# =============================================================================
#essentially combine the steps above to creat prep function for fetching iris data
iris_df = a.get_iris_data()



def prep_iris(iris_df):
    iris_df = iris_df.drop(columns='species_id')  #.drop(columns=column_name)
    iris_df = iris_df.rename(columns={'species_name':'species'})  #.rename({current_col:replace_name})
    
    iris_dummy = pd.get_dummies(iris_df['species'], drop_first=False)   #no drop dummy cols 
    
    iris_df = pd.concat([iris_df, iris_dummy], axis=1)
    
    return iris_df

prep_iris(iris_df).head()
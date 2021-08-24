#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 09:55:17 2021

@author: carolyndavis
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import explore

plt.rcParams['figure.figsize'] = (4, 2)

# =============================================================================
#                             Aquire Titanic data from our mySQL database
# =============================================================================

import env

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic():
    my_query="SELECT * FROM passengers"
    df = pd.read_sql(my_query, get_connection('titanic_db'))
    return df


df = get_titanic()
df.head(2)

# drop rows where age or embarked is null, drop column 'deck', drop passenger_id


# =============================================================================
#                                 Prepare the Titanic data
# =============================================================================

# drop deck > 75% missing and rows where age or embarked is missing
# drop passenger_id
# drop class, as encoded values are in pclass
# create dummy vars & drop sex, embark_town






def prep_titanic(df):
    '''
    take in titanc dataframe, remove all rows where age or embarked is null, 
    get dummy variables for sex and embark_town, 
    and drop sex, deck, passenger_id, class, and embark_town. 
    '''
    df = (
        df[(df.age.notna()) & (df.embarked.notna())].
        drop(columns=['deck', 'passenger_id', 'class']))
    dummy_df = (
        pd.get_dummies(df[['sex', 'embark_town']], prefix=['sex', 'embark']))
    df = (
        pd.concat([df, dummy_df.drop(columns=['sex_male'])], axis=1).
        drop(columns=['sex', 'embark_town']))
    return df





# =============================================================================
#             Split data into Train, Validate, Test
# =============================================================================
def train_validate_test_split(df, target, seed=123):
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


train, validate, test = train_validate_test_split(df, target='survived')
train.head(2)


print(train.shape, validate.shape, test.shape)


#WHAT ARE OUR OBSERVATIONS????????????^^^^
#IMPORTANT QUESTION IN THE EXPLORE STAGE
#TITANIC DATA
#WHAT DOES EACH ROW REPRESENT?
#- IT PREDICTS WHETHER INDIVIDUALS MADE IT OR NOT 


#ALWAYS EXPLORE YOUR TRIANING DATA!!!!!!!
#Build your models on trains but ALSO explore your train
# =============================================================================
#                             UNIVARIATES
# =============================================================================
#First, we need list of categorical variables and one of quantitative variables.


#Univariates look at single variables at a time...AND ITS DISTRIBUTION
cat_vars = ['survived', 'pclass', 'sibsp', 'parch', 'embarked', 'alone', 'sex_female']
quant_vars = ['age', 'fare']

explore.explore_univariate(train, cat_vars, quant_vars)

#First lets verify our lists of variablwes

print(cat_vars)
print(quant_vars)


# We will want to remove the target variable from that list, as the function takes
#  that variable as a separate argument.


cat_vars = cat_vars[1:]

explore.explore_bivariate(train, 'survived', cat_vars, quant_vars)
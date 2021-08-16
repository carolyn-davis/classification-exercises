#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:36:33 2021

@author: carolyndavis
"""

# =============================================================================
#                         Classification Exercises
# =============================================================================
from pydataset import data
import seaborn as sns
import pandas as pd

# =============================================================================
# 1.)  use a python module (pydata or seaborn datasets) containing datasets
#  as a source from the iris data. Create a pandas dataframe, df_iris, from this data.
# =============================================================================
iris_df = data('iris')


# print the first 3 rows

iris_df.head(3)


#print the number of rows and columns (shape)

iris_df.shape
#Ans:  (150, 5)

#print the column names
sorted(iris_df)
#['Petal.Length', 'Petal.Width', 'Sepal.Length', 'Sepal.Width', 'Species']

#print the data type of each column
iris_df.dtypes
# Sepal.Length    float64
# Sepal.Width     float64
# Petal.Length    float64
# Petal.Width     float64
# Species          object\


# =============================================================================
# print the summary statistics for each of the numeric variables.
#  Would you recommend rescaling the data based on these statistics?
# =============================================================================

iris_df.describe()
#No I would not recommend rescaling this df, because the numeric measurments appear to 
#use synonymous measuring units





# =============================================================================
# 2.)Read the Table1_CustDetails table from the Excel_Exercises.xlsx file into a
#  dataframe named df_excel.
# =============================================================================



file_path = "/Users/carolyndavis/Codeup/classification-exercises/"
results_path = "/Users/carolyndavis/Codeup/classification-exercises/"
file_name = "telco_data.xlsx"

def data_loading(path, data_name):
    data = pd.read_excel(path+data_name)        #Loading the data, removing NaNs
    return data.dropna()

telco_data = data_loading(file_path, file_name)

cust_data = telco_data.copy()


cust_data.drop(cust_data.index[101:], inplace=True)

cust_data.iloc[:]


#print the number of rows of your original dataframe
telco_data.shape
#(7032, 12)

#print the first 5 column names
sorted(telco_data)
# ['churn',
#  'contract_type',
#  'customer_id',
#  'dependents',
#  'gender',
#  'internet_service',
#  'is_senior_citizen',
#  'monthly_charges',
#  'partner',
#  'payment_type',
#  'phone_service',
#  'total_charges']


#print the column names that have a data type of object
x = list(telco_data.select_dtypes(['object']).columns)
print(x)
#Ans: ['customer_id', 'gender', 'partner', 'dependents', 'payment_type', 'churn']



#compute the range for each of the numeric variables.

# compute the range for each of the numeric variables.
numeric = telco_data.select_dtypes(include=['float64', 'int64'])


print("Range from min-max of numeric values")
numeric.max() - numeric.min()

# =============================================================================
# is_senior_citizen       1.0
# phone_service           2.0
# internet_service        2.0
# contract_type           2.0
# monthly_charges       100.5
# total_charges        8666.0
# dtype: float64
# =============================================================================





# =============================================================================
# 3.)Read the data from this google sheet into a dataframe, df_google
# =============================================================================

url = 'https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/edit#gid=341089357'

url = url.replace('/edit#gid=', '/export?format=csv&gid=')

gs_df  = pd.read_csv(url)


# print the first 3 rows

gs_df.head(3)



# print the number of rows and column

gs_df.shape
#ANs: (891, 12)

# print the column names
x = list(sorted(gs_df))
# ['Age',
#  'Cabin',
#  'Embarked',
#  'Fare',
#  'Name',
#  'Parch',
#  'PassengerId',
#  'Pclass',
#  'Sex',
#  'SibSp',
#  'Survived',
#  'Ticket']


# print the data type of each column
gs_df.dtypes

# PassengerId      int64
# Survived         int64
# Pclass           int64
# Name            object
# Sex             object
# Age            float64
# SibSp            int64
# Parch            int64
# Ticket          object
# Fare           float64
# Cabin           object
# Embarked        object
# dtype: object




# print the summary statistics for each of the numeric variables

gs_df.describe().T

#              count        mean         std  ...       50%    75%       max
# PassengerId  891.0  446.000000  257.353842  ...  446.0000  668.5  891.0000
# Survived     891.0    0.383838    0.486592  ...    0.0000    1.0    1.0000
# Pclass       891.0    2.308642    0.836071  ...    3.0000    3.0    3.0000
# Age          714.0   29.699118   14.526497  ...   28.0000   38.0   80.0000
# SibSp        891.0    0.523008    1.102743  ...    0.0000    1.0    8.0000
# Parch        891.0    0.381594    0.806057  ...    0.0000    0.0    6.0000
# Fare         891.0   32.204208   49.693429  ...   14.4542   31.0  512.3292

# print the unique values for each of your categorical variables

for column in gs_df.select_dtypes(include='object').columns:
    print(gs_df[column].value_counts())
    
    
    
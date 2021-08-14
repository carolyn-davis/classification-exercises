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
telco_data.dtypes(object)


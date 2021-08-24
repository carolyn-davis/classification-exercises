#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:36:46 2021

@author: carolyndavis
"""

import pandas as pd
import numpy as np
import seaborn as sns
from pydataset import data

import os

# 1.)Make a new python module, acquire.py to hold the following data aquisition functions:
    
# =============================================================================
# Make a function named get_titanic_data that returns the titanic data from the
#  codeup data science database as a pandas data frame. Obtain your data from the
#  Codeup Data Science Database.    l
# =============================================================================
#Gettijng connected to SQL
from env import host, user, password

def get_db_url(db, user, host, password):

    return f'mysql+pymysql://{user}:{password}@{host}/{db}'



##Lets get the titanic data from sQL to a CSV-file/df
def new_titanic_data():

    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup db.
    tit_df = pd.read_sql(sql_query, get_db_url('titanic_db', user, host, password))
    
    return tit_df

def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('tit_df.csv'):
        
        # If csv file exists, read in data from csv file.
        tit_df = pd.read_csv('tit_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        tit_df = new_titanic_data()
        
        # Write DataFrame to a csv file.
        tit_df.to_csv('tit_df.csv')
        
    return tit_df


tit_df = get_titanic_data()

# =============================================================================
# 
# 2.)Make a function named get_iris_data that returns the data from the iris_db on the
#  codeup data science database as a pandas data frame. The returned data frame should
#  include the actual name of the species in addition to the species_ids. Obtain
#  your data from the Codeup Data Science Database.
# =============================================================================

def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
                """
    
    # Read in DataFrame from Codeup db.
    iris_df = pd.read_sql(sql_query, get_db_url('iris_db', user, host, password))
    
    return iris_df


def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        iris_df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        iris_df = new_iris_data()
        
        # Cache data
        iris_df.to_csv('iris_df.csv')
        
    return iris_df

iris_df = get_iris_data()

# =============================================================================
# 3.)Once you've got your get_titanic_data and get_iris_data functions written,
#  now it's time to add caching to them. To do this, edit the beginning of the
#  function to check for a local filename like titanic.csv or iris.csv. If they
#  exist, use the .csv file. If the file doesn't exist, then produce the SQL and
#  pandas necessary to create a dataframe, then write the dataframe to a .csv file
#  with the appropriate name.
# =============================================================================

#See above
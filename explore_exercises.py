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
import scipy.stats as stats

import acquire as a
#went through preparation steps for practice

iris_data = a.get_iris_data()    #data acquired with acquire function

def prep_iris_inline(iris_data):
    iris_data = iris_data.rename(columns={'species_name': 'species'})
    iris_data = iris_data.drop(columns=['species_id'])
    return iris_data

iris_data = prep_iris_inline(iris_data)

#inspect the data 

iris_data.info()


# =============================================================================
#                     # DATA IS CLEAN NOW SPLIT THE DATA
# =============================================================================
from sklearn.model_selection import train_test_split                   
# 20% test, 80% train:

train, test = train_test_split(iris_data, train_size=0.8, random_state=1349, stratify=iris_data.species)

#70/30 train, validate, split

def train_validate_test_split(iris_data, target, seed=123):
    train, test = train_test_split(iris_data, train_size=0.8, random_state=1349, stratify=iris_data.species)
    #80/20 train test split 
    train, validate = train_test_split(train, train_size=0.7, random_state=1349, stratify=train.species)
    #70/30 train validate split
    return train, validate, test

train, validate, test = train_validate_test_split(iris_data, target='species')

train.head()

print(train.shape, validate.shape, test.shape)

# then of the 80% train_validate: 30% validate, 70% train. 

#Output: (84, 5) (36, 5) (30, 5)


# =============================================================================
# 2.)                         UNIVARIATE STATS
# =============================================================================


# =============================================================================
# # For each measurement type (quantitative variable): create a histogram, boxplot,
# #  & compute descriptive statistics (using .describe()).
# =============================================================================

# Create a swarmplot using a melted dataframe of all your numeric variables. 
# The x-axis should be the variable name, the y-axis the measure. Add another 
# dimension using color to represent species. Document takeaways from this visualization.


iris_melted = train.melt(id_vars=['species'])

iris_melted.info()
#observations
#3 cols species, variable, value


plt.rc('font', size=14)   #formatting for swarmplot
plt.rc('figure', figsize=(14, 10))

sns.swarmplot(data=iris_melted, x='variable', y='value', hue='species')



sns.histplot(data=iris_melted,  x='variable', y='value')
sns.boxenplot(data=iris_melted,  x='variable', y='value')
#want each categorical variable to be represented color wise --> use of hue
#separation of petal_length and petal_width seems to be primary indicator of species
#Could location of petal and sepal be usful identifiers?
#virginicas have the biggest petals
#setosas have the smallest petals overall
#Setosa have a range of wide/narrow petals


# =============================================================================
# # For each species (categorical variable): create a frequency table and a bar plot
# #  of those frequencies.
# =============================================================================







# =============================================================================
# # Document takeaways & any actions.
# =============================================================================






# =============================================================================
# 3.)                         BIVARIATE STATS
# =============================================================================
# Visualize each measurement type (y-axis) with the species variable (x-axis) using
#  barplots, adding a horizontal line showing the overall mean of the metric (y-axis).

# For each measurement type, compute the descriptive statistics for each species.

# For virginica & versicolor: Compare the mean petal_width using the Mann-Whitney
#  test (scipy.stats.mannwhitneyu) to see if there is a significant difference between
#  the two groups. Do the same for the other measurement types.

# Document takeaways & any actions.


# 
# Create 4 subplots (2 rows x 2 columns) of scatterplots.

# sepal_length x sepal_width
# petal_length x petal_width
# sepal_area x petal_area
# sepal_length x petal_length


train['sepal_area'] = train.sepal_length * train.sepal_width
train['petal_area'] = train['petal_length'] * train['petal_width']
train.head()

combos = [('sepal_length', 'sepal_width'),
          ('petal_length', 'petal_width'),
          ('sepal_area', 'petal_area'),
          ('sepal_length', 'petal_length')]   #list of tuples for analysis

combos[0]




train.groupby('species')['sepal_area'].sum()
#species
# setosa        473.12
# versicolor    467.81
# virginica     554.13

x = combos[0][0]
y = combos[0][1]

def species_scatter(x, y):
    for species, subset in train.groupby('species'):
        plt.scatter(subset[x], subset[y], label=species)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
species_scatter(x,y)    #this function tests the data with one plot 

#Plot for each measured feature 
plt.subplot(2,2,1)
species_scatter(combos[0][0], combos[0][1])
plt.subplot(2,2,2)
species_scatter(combos[1][0], combos[1][1])
plt.subplot(2,2,3)
species_scatter(combos[2][0], combos[2][1])
plt.subplot(2,2,4)
species_scatter(combos[3][0], combos[3][1])
plt.tight_layout()

#sepal_length is measurable/useful for comparison
#petal_area appears very indicative of species type
#there isnt a lot of variation/uniqueness for sepal_area


for i, pair in enumerate(combos):
    plt.subplot(2,2,i+1)
    species_scatter(pair[0], pair[1])

# =============================================================================
# 4.)                         MULTIVARIATE STATS 
# =============================================================================
# Visualize the interaction of each measurement type with the others using a pairplot
#  (or scatter matrix or something similar) and add color to represent species.

# Create a swarmplot using a melted dataframe of all your numeric variables.
#  The x-axis should be the variable name, the y-axis the measure. Add another
#  dimension using color to represent species. Document takeaways from this visualization.

# Ask a specific question of the data, such as: is the sepal area signficantly
#  different in virginica compared to setosa? Answer the question through both a plot
#  and using a mann-whitney or t-test. If you use a t-test, be sure assumptions are
#  met (independence, normality, equal variance).

# Document takeaways and any actions.
#---------------------------------------

# Create a heatmap of each variable layering correlation coefficient on top

train.corr()

sns.heatmap(train.corr(), cmap='mako', center=0, annot=True)
plt.show()

#observations:
# most measures correlate with each other, the least is sepal_width with sepal_lenght
#  @ -.15.

# Negative correlations: Wider sepals => shorter & narrower petals (smaller petal areas)

# Positive correlations: Longer sepals => longer & wider petals (larger petal areas)

# Little to no LINEAR correlation: sepal length & sepal width



# =============================================================================
# # Visualize the interaction of each measurement type with the others using a pairplot
# #  (or scatter matrix or something similar) and add color to represent species.
# =============================================================================


                                #Creating a Scatter Matrix
pd.plotting.scatter_matrix(train)
plt.show()

                                #Creating the Pairplot 
sns.pairplot(train, hue='species')
plt.show()

train.info()

            #Observations
# petal length + petal width show the most seperation between species.
# setosa has the shortest and narrowest petals. It will be the easiest to determine.
# petal area seems to show the largest separation between virginica & versicolor
#  of all the individual features.
# virginica shows slightly longer sepals, but whether that difference is significant,
#  it's hard to say.
# virginica and versicolor show little to no difference when it comes to the width of the
#  sepals.



# =============================================================================
# # Ask a specific question of the data, such as: is the sepal area signficantly
# #  different in virginica compared to setosa? Answer the question through both a plot
# #  and using a mann-whitney or t-test. If you use a t-test, be sure assumptions are
# #  met (independence, normality, equal variance).
# =============================================================================
#Driving Question: Is sepal length extensively different in virginica species when 
    #when compared to veriscolor?
#Null Hyp: Sepal length is the same in virginica and versicolor.
#Alt Hyp: Sepal length is very different in virginica and versicolor.
#alpha = .05



virginica = train[train['species'] == 'virginica']
versicolor = train[train['species'] == 'versicolor']

#Visualize the Data for Comparison
    
virginica.hist()
plt.show()

versicolor.hist()   #much bigger
plt.show()

virginica.describe()['sepal_length']['std'] #dcalling on features with in dataset
#0.532985998009398
versicolor.describe()['sepal_length']
# count    28.000000
# mean      5.935714
# std       0.512231
# min       5.000000
# 25%       5.600000
# 50%       6.000000
# 75%       6.300000
# max       6.900000
# Name: sepal_length, dtype: float64

tstat, p = stats.ttest_ind(virginica.sepal_length,
                           versicolor.sepal_length,
                           equal_var=False)

print(tstat)

#tstat = 5.11297615624047

print(p)
#p_value = 4.304267066877367e-06

#Concluson.: Null hypothesis is rejected that sepal length is the same for virginica and 
#           versicolor.
#Sepal length is a measurable and considered useful feature for comparison/analysis


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 15:00:08 2021

@author: carolyndavis
"""

# =============================================================================
# 1.)                                 Attendance Data
# =============================================================================
# Load the attendance.csv file and calculate an attendnace percentage for each student.
# One half day is worth 50% of a full day, and 10 tardies is equal to one absence.


import pandas as pd
import seaborn as sns


attend_df = pd.read_csv('untidy-data/attendance.csv')


#Need to melt dataframe from wide to long

attend_melt = attend_df.melt(id_vars='Unnamed: 0',var_name = 'day', value_name='grade') #shifted frame from wide to long


#Lets rename the columns


attend_melt.columns = ['student', 'day', 'grade']  


#converting letter grades to numbers for average
def letter_grade(a):
    if a == 'P':
        return 1
    elif a == 'H':
        return 0.5
    elif a == 'T':
        return 0.9
    else:
        return 0
#REMEMBER THOSE BRACKETS

attend_melt['num_grade'] = attend_melt['grade'].apply(letter_grade)

attend_melt.groupby('student').num_grade.mean()    #groupby calculates mean grade of students


# student
# Billy    0.5250
# Jane     0.6875
# John     0.9125
# Sally    0.7625
# Name: num_grade, dtype: float64


# =============================================================================
# 2.)                                     Coffee Levels
# =============================================================================
# Read the coffee_levels.csv file.

coffee_df = pd.read_csv('untidy-data/coffee_levels.csv')

#Looking at the data:
sns.lineplot(x = 'hour',
             y = 'coffee_amount',
             data = coffee_df,
             hue = 'coffee_carafe')


# Transform the data so that each carafe is in it's own column.

coffee_pivot = coffee_df.pivot_table(index = ['hour'],
                                     columns = 'coffee_carafe',
                                     values = 'coffee_amount')

coffee_pivot.plot()



# Is this the best shape for the data?

#Ans: depends I personally find the pivot table more comprehendable when compared to the lineplot




# =============================================================================
# 3.)                                     Cake Recipes
# =============================================================================



# a.)Read the cake_recipes.csv data. This data set contains cake tastiness scores
#     for combinations of different recipes, oven rack positions, and oven temperatures.

cake_df = pd.read_csv('untidy-data/cake_recipes.csv')


# b.)Tidy the data as necessary.
#numeric values for column names, multiple variables in rows 

cake_df.columns

#Index(['recipe:position', '225', '250', '275', '300'], dtype='object')


cake_df['recipe:position'].str.split(":", expand = True)   # splits the 'recipe:position' column

#expand and create new columns

cake_df[['recipe', 'position']] = cake_df['recipe:position'].str.split(':', expand = True)

cake_df.drop(columns = 'recipe:position', inplace = True)   #This drops the unsplit cols 

#Make data from wide to long: MELT

cake_melt = cake_df.melt(id_vars = ['recipe', 'position'], var_name = 'temperature', value_name = 'score')



# c.)Which recipe, on average, is the best? recipe b
#groupby for melt

cake_melt.groupby(['recipe']).score.mean().idxmax()

#Ans: recipe b on average is the best



# d.)Which oven temperature, on average, produces the best results? 275

cake_melt.groupby('temperature').score.mean().idxmax()
#Ans: best temp on average is 275 F

# e.)Which combination of recipe, rack position, and temperature gives the best result?
#     recipe b, bottom rack, 300 degrees

cake_melt.groupby(['temperature', 'recipe', 'position']).score.mean()



top_comb = cake_melt.groupby(['temperature', 'recipe', 'position']).score.mean().idxmax()
top_score = cake_melt.groupby(['temperature', 'recipe', 'position']).score.mean().max()

print(top_comb)
print('-' * 80)
print(top_score)

#Ans: best combination = ('300', 'b', 'bottom')
#       top_score = 99.2485405378462
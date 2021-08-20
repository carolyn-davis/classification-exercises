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
animal_df = pd.DataFrame({
    'actual': [46, 13],
    'prediction': [7, 34]})

pd.crosstab(animal_df.actual, animal_df.prediction)

confusion_matrix(animal_df.actual, animal_df.prediction,
                 labels = (0,1))

# =============================================================================
# 2.)Given the following confusion matrix, evaluate (by hand) the model's performance.
# =============================================================================


# In the context of this problem, what is a false positive?
#Says there are 7 cats but there are actually 13



# In the context of this problem, what is a false negative?



# How would you describe this model?

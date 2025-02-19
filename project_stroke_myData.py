#!/usr/bin/env python3
# coding: utf-8

__author__ = "Joanne ADAM"
__license__ = "GNU General Public License"
__version__ = "1.0"
__email__ = "joanne.adam@edu.dsti.institute"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import cross_val_score

from IPython.display import display # to display like in jupyter-noteboook

from project_ML_DSTI import stroke_prediction

# =======================================================================================
show = False

# =======================================================================================
# ---------------------------------------------------------------------------------------
# --- MODELING --------------------------------------------------------------------------
sp = stroke_prediction(filename="stroke_data_afterFeatureEngineering.csv")
sp.split_dataset_train_test() # Get the training/testing datasets (size=0.2)

# Over-sampling using SMOTE
sp.oversampling() 

#sp.create_myData() # always 0
sp.create_DataPapy() # LR:1, RF:0, MLP:1, DT:0

# --------------------------------------------------------
print("-------------------------------------------------")
print("--- Logistic Regression -------------------------") 
sp.do_logisticRegression() # --> Get very bad solution because undersampling of 0s
sp.do_model_evaluation(smote=True,  display=False) # Model evaluation:
sp.model_myData()

# --------------------------------------------------------
print("-------------------------------------------------")
print("--- Random Forest -------------------------------") 
sp.do_randomForest() # --> Get very bad solution because undersampling of 0s
sp.do_model_evaluation(smote=True , display=False) # Model evaluation:
sp.model_myData()

# --------------------------------------------------------
print("-------------------------------------------------")
print("--- Multi-Layer Perceptron ----------------------") 
# --- Very long
sp.do_MLP() # --> Get very bad solution because undersampling of 0s
sp.do_model_evaluation(smote=True , display=False) # Model evaluation:
sp.model_myData()

# --------------------------------------------------------
print("------------------------------------------------")
print("--- Decision Tree Classifier -------------------")
sp.do_decisionTreeClassifier() # --> Get very bad solution because undersampling of 0s
sp.do_model_evaluation(smote=True , display=False) # Model evaluation:
sp.model_myData()

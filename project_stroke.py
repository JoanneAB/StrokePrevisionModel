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

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import cross_val_score

from IPython.display import display # to display like in jupyter-noteboook

from project_ML_DSTI import stroke_prediction

# =======================================================================================
show = False

# =======================================================================================


# ---------------------------------------------------------------------------------------
# --- DATA ANALYSIS ---------------------------------------------------------------------
sp = stroke_prediction()
sp.do_data_analysis()

# ---------------------------------------------------------------------------------------
# --- FEATURE ENGINERING ----------------------------------------------------------------
sp = stroke_prediction(filename="stroke_data_afterDataAnalysis.csv")
sp.do_feature_engineering()

# ---------------------------------------------------------------------------------------
# --- MODELING --------------------------------------------------------------------------

sp = stroke_prediction(filename="stroke_data_afterFeatureEngineering.csv")

# We split the dataset on several parts:
# -- training set: 
# . self.x_train : data for training
# . self.y_train : solution for training
# -- testing set:
# . self.x_test : data for testing
# . self.y_test : solution for testing
# And get the models stored in the parameters:
# . self.model       : parameters of the model
# . self.y_predicted : prediction of the solution

# --------------------------------------------------------
sp.split_dataset_train_test() # Get the training/testing datasets (size=0.2)

# Over-sampling using SMOTE
# -> get self.x_train_smote, self.y_train_smote, self.train_smote
sp.oversampling() 
#sp.df = pd.concat([sp.train_smote, sp.df_test])

# --------------------------------------------------------
print("-------------------------------------------------")
print("--- Logistic Regression -------------------------") 
sp.do_logisticRegression() # --> Get very bad solution because undersampling of 0s
sp.do_model_evaluation(smote=True,  display=False) # Model evaluation:
sp.do_model_evaluation(smote=False, display=False) # Model evaluation:

# --------------------------------------------------------
print("-------------------------------------------------")
print("--- Random Forest -------------------------------") 
sp.do_randomForest() # --> Get very bad solution because undersampling of 0s
sp.do_model_evaluation(smote=True , display=False) # Model evaluation:
sp.do_model_evaluation(smote=False, display=False) # Model evaluation:

# --------------------------------------------------------
print("-------------------------------------------------")
print("--- Multi-Layer Perceptron ----------------------") 
# --- Very long
sp.do_MLP() # --> Get very bad solution because undersampling of 0s
sp.do_model_evaluation(smote=True , display=False) # Model evaluation:
sp.do_model_evaluation(smote=False, display=False) # Model evaluation:

# --------------------------------------------------------
print("------------------------------------------------")
print("--- Decision Tree Classifier -------------------")
sp.do_decisionTreeClassifier() # --> Get very bad solution because undersampling of 0s
sp.do_model_evaluation(smote=True , display=False) # Model evaluation:
sp.do_model_evaluation(smote=False, display=False) # Model evaluation:

## --------------------------------------------------------
#print("------------------------------------------------")
#print("--- SVM ----------------------------------------")
#sp.do_svm() # --> Get very bad solution because undersampling of 0s
#sp.do_model_evaluation(smote=True , display=False) # Model evaluation:
#sp.do_model_evaluation(smote=False, display=False) # Model evaluation:

# Save the variables into a file for later use: 
with open('project.pckl', 'wb') as fic:
  pickle.dump(sp, fic)
exit()


sp.plot_ageBMIAvgGlucoseLevel_stroke01(text="over")
sp.plot_correlationMatrix(filename="correlationMatrix_over.png", columns=sp.columns)

## Get the variables from the file so no need to compute long one:
#with open('project.pckl', 'rb') as fic:
#  sp = pickle.load(fic)

# --------------------------------------------------------
# --------------------------------------------------------
sp.plot_boxplot_metrics()
for model in sp.results.keys():
  for smote in [0,1]:
    print("---", model, smote)
    print(sp.results[model]['smote_%d'%smote]['process_time'])


sp.plot_metrics_all()

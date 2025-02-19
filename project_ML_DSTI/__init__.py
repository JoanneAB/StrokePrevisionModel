#!/usr/bin/env python3
# coding: utf-8

__author__ = "Joanne ADAM"
__license__ = "GNU General Public License"
__version__ = "1.0"
__email__ = "joanne.adam@edu.dsti.institute"

import numpy as np
import pandas as pd

class stroke_prediction(object):
  """
  """
  from ._plot_functions import plot_correlationMatrix, plot_hist_stack, plot_pie, plot_hist, plot_hist_boxplot, plot_var1_bmi_missing, plot_boxplot_metrics, plot_metrics_all, plot_ageBMIAvgGlucoseLevel_stroke01
  from ._data_handling import load_data, rename_columns, save_data, create_myData, create_DataPapy
  from ._data_analysis import do_data_analysis
  from ._feature_engineering import get_dummies, scale_numerics, do_feature_engineering
  from ._modeling import split_dataset_train_test, oversampling, do_logisticRegression, do_model_evaluation, do_randomForest, do_MLP, do_svm, do_decisionTreeClassifier, model_myData

  # --------------------------------------------------------------------------
  def __init__(self, filename="stroke_data.csv", path="/home/joanne/perso/Rokkasho/DSTI/lessons/module_02/machineLearningPython/project"):
    """
    """
    self.path = path
    self.filename = filename
   
    self.df = None
    self.df_0 = None
    self.df_1 = None

    self.columns = []

    # Dictionary to save the result of all the modelings:
    self.results = {}

    # Load the data from the file
    self.load_data()

  # --------------------------------------------------------------------------

#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------------------------------------------------------------------
def load_data(self, sep=",", index_col="id"):
  """
  Load the data from file
  """
  self.df = pd.read_csv(self.path+'/'+self.filename, sep=sep, index_col=index_col)


# ---------------------------------------------------------------------------------------
def rename_columns(self, columns, inplace=True):
  """
  Rename the name of one or several columns in the dataset
  """
  self.df.rename(columns=columns, inplace=inplace)


# ---------------------------------------------------------------------------------------
def save_data(self, filename):
  """
  Save the dataset into a file
  """

# ---------------------------------------------------------------------------------------
def create_myData(self):
  """
  Create a testing set with specific data
  """
  self.myData = pd.DataFrame(columns=self.df.columns[self.df.columns != 'stroke'])
  self.myData.loc[0, 'gender_female'] = 1
  self.myData.loc[0, 'age']           = 0.487 # 40 year old
  self.myData.loc[0, 'hypertension']  = 0
  self.myData.loc[0, 'heart_disease'] = 0
  self.myData.loc[0, 'ever_married']  = 1
  self.myData.loc[0, 'avg_glucose_level'] = 0.24 # mean value of original dataset
  self.myData.loc[0, 'bmi']               = 0.255 # BMI=27.6 
  self.myData.loc[0, 'job_government']    = 0
  self.myData.loc[0, 'job_never']         = 0
  self.myData.loc[0, 'job_private']       = 1
  self.myData.loc[0, 'job_selfemployed']  = 0
  self.myData.loc[0, 'job_children']      = 0
  self.myData.loc[0, 'smoker_unknown']    = 0
  self.myData.loc[0, 'smoker_formerly']   = 0
  self.myData.loc[0, 'smoker_never']      = 1
  self.myData.loc[0, 'smoker']            = 0

# ---------------------------------------------------------------------------------------
def create_DataPapy(self):
  """
  Create a testing set with specific data
  """
  self.myData = pd.DataFrame(columns=self.df.columns[self.df.columns != 'stroke'])
  self.myData.loc[0, 'gender_female'] = 0
  self.myData.loc[0, 'age']           = 1.0 # 85 year old
  self.myData.loc[0, 'hypertension']  = 1
  self.myData.loc[0, 'heart_disease'] = 0
  self.myData.loc[0, 'ever_married']  = 1
  self.myData.loc[0, 'avg_glucose_level'] = 0.24 # mean value of original dataset
  self.myData.loc[0, 'bmi']               = 0.173 # BMI=22.0 
  self.myData.loc[0, 'job_government']    = 1
  self.myData.loc[0, 'job_never']         = 0
  self.myData.loc[0, 'job_private']       = 0
  self.myData.loc[0, 'job_selfemployed']  = 0
  self.myData.loc[0, 'job_children']      = 0
  self.myData.loc[0, 'smoker_unknown']    = 0
  self.myData.loc[0, 'smoker_formerly']   = 0
  self.myData.loc[0, 'smoker_never']      = 1
  self.myData.loc[0, 'smoker']            = 0

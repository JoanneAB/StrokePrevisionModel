#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from IPython.display import display # to display like in jupyter-noteboook


# ---------------------------------------------------------------------------------------
def do_data_analysis(self):
  """
  Inspection of the dataset
  """

  # Look at the dataset:
  display(self.df.head(10))
  display(self.df.tail(10))

  display(self.df.describe())

  # Look at the total number for each value :
  # work_type:
  print("Values and frequency for the attribute 'work_type':")
  display(self.df.work_type.unique())
  print([self.df.work_type[self.df.work_type=="Private"].count(),
    self.df.work_type[self.df.work_type=="Self-employed"].count(),
    self.df.work_type[self.df.work_type=="Govt_job"].count(),
    self.df.work_type[self.df.work_type=="children"].count(),
    self.df.work_type[self.df.work_type=="Never_worked"].count()])

  # hypertension:
  print("Frequency for the attribute 'hypertension':")
  print([self.df.hypertension[self.df.hypertension==0].count(),
         self.df.hypertension[self.df.hypertension==1].count()])

  # heart-disease:
  print("Frequency for the attribute 'heart_disease':")
  print([self.df.heart_disease[self.df.heart_disease==0].count(),
         self.df.heart_disease[self.df.heart_disease==1].count()])

  # -----------------------------------------
  # --- Look at distribution of data:
  self.plot_pie("hypertension")
  self.plot_pie("heart_disease")
  self.plot_pie("stroke")

  self.plot_correlationMatrix(columns=["stroke", "heart_disease", "hypertension"], filename="correlationMatrix_strokeHeartDiseaseHypertension.png", figsize=(6,5), fmt='0.3f', annot_fontsize=15)

  self.plot_hist("age", bins=250)
  self.plot_hist("avg_glucose_level", bins=100)
  self.plot_hist("bmi", bins=100)

  self.plot_hist_boxplot("age", bins=20)
  self.plot_hist_boxplot("avg_glucose_level", bins=20)
  self.plot_hist_boxplot("bmi", bins=20)


  # -----------------------------------------
  # --- Outliers values of BMI:
  display(self.df.bmi.sort_values(ascending=False).head(10))

  # -----------------------------------------
  # --- See if obvious relations in data.
  # gender vs. stroke:
  self.plot_hist_stack(x="gender", hue="stroke")

  # hypertension vs. stroke:
  self.plot_hist_stack(x="hypertension", hue="stroke")

  # heart_disease vs. stroke:
  self.plot_hist_stack(x="heart_disease", hue="stroke")

  # -----------------------------------------
  # --- Where are the missing values:
  display(self.df.isna().sum())

  display(len(self.df))
  display(self.df.columns)

  # -----------------------------------------
  # all column names start with lower-case except Residence_type -> change it:
  self.rename_columns({"Residence_type":"residence_type"})

  # =======================================================================================
  # =======================================================================================
  self.df.to_csv("stroke_data_afterDataAnalysis.csv", sep=',', encoding='utf-8')


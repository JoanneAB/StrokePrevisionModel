#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display # to display like in jupyter-noteboook

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def get_dummies(self, attribute):
  """
  """
  self.df.loc[:, [attribute]] = pd.get_dummies(self.df[attribute], drop_first=True).values

# ---------------------------------------------------------------------------------------
def scale_numerics(self, column):
  """
  Scale the numerical values for the attributes to be in the range [0-1]
  """
  mini = np.min(self.df[column])
  maxi = np.max(self.df[column])

  self.df[column] = (self.df[column] - mini) / (maxi - mini)


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def do_feature_engineering(self):

  # ---------------------------------------------------------------------------------------
  # Change Ever_married : No/Yes -> 0/1
  self.get_dummies("ever_married")

  # ---------------------------------------------------------------------------------------
  # Change residence_type : rural/urban -> 0/1
  self.get_dummies("residence_type")
  self.rename_columns({"residence_type":"urban_residence"})

  # ---------------------------------------------------------------------------------------
  # Gender is Male/Female/Other. 
  # Only one 'Other' for id=56156. This person did not have a stroke. 
  # The size of the population of the person who did not have a stroke is very large (4 861) 
  # compared to the population of the person who did have a stroke (249). 
  # We can discard this only data as it might not affect much the result. 
#  display(self.df.loc[self.df.gender=='Other', 'stroke'])
  self.df = self.df.drop(index=self.df.loc[self.df.gender=='Other',:].index)

  # ---------------------------------------------------------------------------------------
  # ---------------------------------------------------------------------------------------
  # Change gender : Male/Female -> 0/1:
  self.get_dummies("gender")
  self.rename_columns({"gender":"gender_female"})

  # ---------------------------------------------------------------------------------------
  # ---------------------------------------------------------------------------------------
  # Create new columns for the work_type:
  self.df = pd.concat([self.df, pd.get_dummies(self.df.work_type)],axis=1).drop(columns="work_type")
  self.rename_columns({"Private":"job_private", "Govt_job":"job_government", "Never_worked":"job_never", "children":"job_children", "Self-employed":"job_selfemployed"})
  
  # ---------------------------------------------------------------------------------------
  # ---------------------------------------------------------------------------------------
  # Create new columns for the smoking status ?
  #    Never smoked : 1892
  #         Unknown : 1544
  # formerly smoked : 884
  #          smokes : 789
#  print(self.df.smoking_status.unique())
  # We have two choices : 
  # 1. ordinal encoding: unknown->nan ; never->1 ; formerly->2 ; smokes-> 3
  # 2.  vector encoding: one column per value

  self.df_svg = self.df.copy()

  self.df['smoking_status'] = self.df['smoking_status'].replace(['Unknown']        , np.nan)
  self.df['smoking_status'] = self.df['smoking_status'].replace(['never smoked']   , 1)
  self.df['smoking_status'] = self.df['smoking_status'].replace(['formerly smoked'], 2)
  self.df['smoking_status'] = self.df['smoking_status'].replace(['smokes']         , 3)

  # Check the correlation matrix if we see significant relation between the smoking_status and the rest:
  self.plot_correlationMatrix(filename="correlationMatrix_smokingOrdinalEncoding.png") #, title="Correlation matrix for ordinal encoding of smoking_status")
  # --> nothing seem to appear.

  # Try vector encoding:
  self.df = self.df_svg.copy()
  # if keep Unknown as a valid value... FIXME`
  self.df = pd.concat([self.df, pd.get_dummies(self.df.smoking_status)],axis=1).drop(columns="smoking_status")
  self.rename_columns({"Unknown":"smoker_unknown", 
                "never smoked":"smoker_never", 
             "formerly smoked":"smoker_formerly", 
                      "smokes":"smoker"})

  self.plot_correlationMatrix(filename="correlationMatrix_smokingVectorEncoding.png") #, title="Correlation matrix for vector encoding of smoking_status")
  # --> some correlation appear and the size of the dataset is not much bigger, we keep this configuration.

  # TODO/FIXME what to do with the missing values ???

  # ---------------------------------------------------------------------------------------
  # ---------------------------------------------------------------------------------------
  # -- BMI
  # Remove the two entries with bmi>90
  self.df = self.df.drop(index=self.df.loc[self.df.bmi>90,:].index)

  # --- Searches on BMI to replace missing values
  self.df.bmi.hist(bins=50, color='0')

  fig, ax = plt.subplots(1, figsize=(20, 10))
  ax = self.df.age.hist(bins=50, color='0', ax=ax)

  # job_children vs. BMI
  # Not usefull because bolean values of job_children.
  self.plot_var1_bmi_missing("job_children")

  # age vs. BMI
  self.plot_var1_bmi_missing("age")
  
  # Values are discrete and integers for age >= 2
  self.plot_var1_bmi_missing("age", xlim=[0, 10], ylim=[0, 40], title_txt="zoom")
  
  # Search for the average bmi for each discrete age: 
  # discrete age for age >=2:
  #avg_bmi_per_age_ge2 = df[df.age>=2].groupby(["age"]).mean().bmi
  #
  ## get the average for age <2:
  #avg_bmi_per_age_lt2 = np.mean(df[df.age<2].bmi)
  #
  #ages = list(avg_bmi_per_age_ge2.index)
  #bmi_per_age = avg_bmi_per_age_ge2.value
  
  # Update BMI to remove missing values:
  avg_bmi_per_age = self.df.groupby(["age"]).mean().bmi
  
  ages = list(avg_bmi_per_age.index)
  bmi_per_age = avg_bmi_per_age.values
  
  # Create a dictionary to link each age to a average BMI:
  testi = {}
  for k in range(len(ages)):
    testi[ages[k]] = bmi_per_age[k]
  
  # Replace the missing BMI:
  for k in self.df.loc[self.df.bmi.isna()].index:
    self.df.loc[k, "bmi"] = testi[self.df.loc[k, "age"]]

  plt.figure(figsize=(14,5))
  plt.scatter(self.df.age, self.df.bmi, marker='+', color='0', s=150)
  plt.scatter(ages, bmi_per_age ,marker='o', color='red', s=100)
  plt.xlabel("Age"); plt.ylabel("BMI")
  plt.xlim(2.5, 40.5); plt.ylim(10,65)
  plt.legend([": Data", ": Average BMI per age"])
  plt.savefig("%s/figures/plot_age_bmi_replaced.png"%self.path, bbox_inches='tight') 
  

  # Check all missing values are replaced :
#  display(self.df.bmi.isna().sum())
#  display(self.df.isna().sum())
  
  # ---------------------------------------------------------------------------------------
  # Very small correlation for 'urban_residence', 'job_never' and 'gender_female' -> remove ??
  self.df = self.df.drop(columns="urban_residence") #, "job_never", "gender_female"])

  # ---------------------------------------------------------------------------------------
 
  # Plot of the distribution of the 0/1 of stroke:
  self.plot_ageBMIAvgGlucoseLevel_stroke01()

  # Scale the numerical columns to be in the range [0-1]:
#  display(self.df[["age", "bmi", "avg_glucose_level"]].describe())
  self.scale_numerics('age')
  self.scale_numerics('bmi')
  self.scale_numerics('avg_glucose_level')
#  display(self.df[["age", "bmi", "avg_glucose_level"]].describe())

  # =======================================================================================
  # Save the columns to have correlation plots with same order always:
  self.columns = list(self.df.columns)
 
  # =======================================================================================
  # ----- Plot the correlation matrix :
  self.plot_correlationMatrix(filename="correlationMatrix_final.png", columns=self.columns)
  
  # =======================================================================================
  # =======================================================================================
  self.df.to_csv("stroke_data_afterFeatureEngineering.csv", sep=',', encoding='utf-8')

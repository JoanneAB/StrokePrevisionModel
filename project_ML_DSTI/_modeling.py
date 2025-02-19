#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import process_time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, classification_report
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

from IPython.display import display # to display like in jupyter-noteboook

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def split_dataset_train_test(self, n_splits=5, size=0.2):
  """
  Get the training and testing datasets from the original dataset.

  n_splits = 1 : do not do k-fold
           > 1 : d k-fold with k=n_splits
  """

  self.x = self.df.loc[:, self.df.columns != 'stroke'] # data
  self.y = self.df.loc[:, self.df.columns == 'stroke'] # target

  # Store all the sets:
  self.x_train = []
  self.y_train = []
  self.x_test  = []
  self.y_test  = []
 
  if n_splits == 1:
    # No k-fold -> do regular spliting:
    train, test = train_test_split(self.df, test_size=size, random_state=0)

    # Values for the training data:
    self.x_train.append( self.df_train.loc[:, self.df_train.columns != 'stroke'])
    self.y_train.append( self.df_train.loc[:, self.df_train.columns == 'stroke'])

    # Values for the test data:
    self.x_test.append( self.df_test.loc[:, self.df_test.columns != 'stroke'])
    self.y_test.append( self.df_test.loc[:, self.df_test.columns == 'stroke'])

  else:
    # k-fold spliting:
    strat_kfold = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in strat_kfold.split(self.x, self.y):
      # Values for the training data:
      self.x_train.append( self.x.iloc[train_index,:])
      self.y_train.append( self.y.iloc[train_index,:])

      # Values for the test data:
      self.x_test.append( self.x.iloc[test_index,:])
      self.y_test.append( self.y.iloc[test_index,:])

# ---------------------------------------------------------------------------------------
def oversampling(self, sampling='minority', random_state=0):
  """
  Oversampling of the dataset using SMOTE
  Over-write the dataset.
  """

  smote = SMOTE(sampling_strategy=sampling, random_state=random_state)

  self.x_train_smote = []
  self.y_train_smote = []
  self.train_smote = []

  for k in range(len(self.x_train)):
    x_train_smote, y_train_smote = smote.fit_resample(self.x_train[k], self.y_train[k])
    self.x_train_smote.append(x_train_smote)
    self.y_train_smote.append(y_train_smote)

    self.train_smote.append( pd.concat([x_train_smote, y_train_smote], axis=1))

# ---------------------------------------------------------------------------------------
def do_logisticRegression(self, max_iter=1000):
  """
  Apply logistic regression.
  Well suited for classification (our case).
  """
  self.model = LogisticRegression(random_state=0, max_iter=max_iter) 

# ---------------------------------------------------------------------------------------
def do_randomForest(self):
  """
  Apply Random Forest algorithm.
  """
  self.model = RandomForestClassifier(random_state=0)

# ---------------------------------------------------------------------------------------
def do_MLP(self):
  """
  Apply Multi-layer Perceptron Classifier algorithm.
  """
  self.model = MLPClassifier(hidden_layer_sizes=(20,10), max_iter=1000, random_state=0)

# ---------------------------------------------------------------------------------------
def do_decisionTreeClassifier(self):
  """
  Apply Decicion Tree Classifier algorithm
  """
  self.model = DecisionTreeClassifier(random_state=0)

# ---------------------------------------------------------------------------------------
def do_svm(self):
  """
  Rank Based SVM

  New modeling to try to improve recall scores 
  """
  self.model = SVC(C=1.0, kernel='linear', class_weight='balanced')

# ---------------------------------------------------------------------------------------
def do_model_evaluation(self, smote=True, save_tree=False, display=False):
  """
  Check how good the model is.
  """

  model_name = str(self.model.__str__()).split('(')[0]

  # Create the results dictionary if does not exist to save the results of many modelings:
  if model_name not in self.results.keys():
    self.results[model_name] = {}
    self.results[model_name]['smote_1'] = {}
    self.results[model_name]['smote_0'] = {}

  if smote:
    self.y_predicted_smote = [] # Save all predicted data
    self.cf_smote = []
    self.metrics_smote = pd.DataFrame(columns=["Recall", "Precision", "F1-score", "Accuracy"])
    self.report_smote  = [] # save all reports
  else:
    self.y_predicted = [] # Save all predicted data
    self.cf = []
    self.metrics = pd.DataFrame(columns=["Recall", "Precision", "F1-score", "Accuracy"])
    self.report  = [] # save all reports

  for k in range(len(self.x_train)):
    if smote:
      t0 = process_time()
      clf = self.model.fit(X=self.x_train_smote[k], y=self.y_train_smote[k].values.ravel())
      t1 = process_time()

      self.results[model_name]['smote_1']['process_time'] = t1-t0 # Save the processing time

      # Save the decision tree into a file:
      if save_tree:
        dot_data = export_graphviz(clf, out_file="decisionTreeClassifier.dot", filled=True, rounded=True)

      # Get the predicted model:
      self.y_predicted_smote.append( self.model.predict(self.x_test[k]))

      # Compute the confusion matrix:
      self.cf_smote.append( pd.DataFrame(columns=["data_0","data_1"],index=["model_0","model_1"]))
      self.cf_smote[k].loc[:,:] = confusion_matrix(y_true=self.y_test[k], y_pred=self.y_predicted_smote[k])/len(self.y_test[k])

      #  Compute smote metrics:
      self.metrics_smote.loc[k, "Recall"] = recall_score(y_true=self.y_test[k], y_pred=self.y_predicted_smote[k], zero_division=0)
      self.metrics_smote.loc[k, "Precision"] = precision_score(y_true=self.y_test[k], y_pred=self.y_predicted_smote[k], zero_division=0)
      self.metrics_smote.loc[k, "F1-score"]  = f1_score(y_true=self.y_test[k], y_pred=self.y_predicted_smote[k], zero_division=0)
      self.metrics_smote.loc[k, "Accuracy"]  = accuracy_score(y_true=self.y_test[k], y_pred=self.y_predicted_smote[k])

      # Save the results in the dictionary of results of all modelings:
      self.results[model_name]['smote_1']['metrics'] = self.metrics_smote
      self.results[model_name]['smote_1']['confusion_matrix'] = self.cf_smote

      # Get smote report:
      self.report_smote.append( classification_report(y_true=self.y_test[k], y_pred=self.y_predicted_smote[k], zero_division=0))

      if display:
        print('-- smote -------------------------------------')
        print(self.metrics_smote)

    # -----------
    else: # no smote:
      t0 = process_time()
      clf = self.model.fit(X=self.x_train[k], y=self.y_train[k].values.ravel())
      t1 = process_time()
      self.results[model_name]['smote_0']['process_time'] = t1-t0 # Save the processing time

      # Save the decision tree into a file:
      if save_tree:
        dot_data = export_graphviz(clf, out_file="decisionTreeClassifier.dot", filled=True, rounded=True)  

      # Get the predicted model:
      self.y_predicted.append( self.model.predict(self.x_test[k]))

      # Compute the confusion matrix:
      self.cf.append( pd.DataFrame(columns=["data_0","data_1"],index=["model_0","model_1"]))
      self.cf[k].loc[:,:] = confusion_matrix(y_true=self.y_test[k], y_pred=self.y_predicted[k])/len(self.y_test[k])

      #  Compute metrics:
      self.metrics.loc[k, "Recall"]    = recall_score(y_true=self.y_test[k], y_pred=self.y_predicted[k], zero_division=0)
      self.metrics.loc[k, "Precision"] = precision_score(y_true=self.y_test[k], y_pred=self.y_predicted[k], zero_division=0)
      self.metrics.loc[k, "F1-score"]  = f1_score(y_true=self.y_test[k], y_pred=self.y_predicted[k], zero_division=0)
      self.metrics.loc[k, "Accuracy"]  = accuracy_score(y_true=self.y_test[k], y_pred=self.y_predicted_smote[k])

      # Save the results in the dictionary of results of all modelings:
      self.results[model_name]['smote_0']['metrics'] = self.metrics
      self.results[model_name]['smote_0']['confusion_matrix'] = self.cf

      self.report.append( classification_report(y_true=self.y_test[k], y_pred=self.y_predicted[k], zero_division=0))

      if display:
        print('-- no smote ----------------------------------')
        print(self.metrics)

# ---------------------------------------------------------------------------------------
def model_myData(self):
  """
  Test with my data
  """
  self.y_predicted_myData = self.model.predict(self.myData)
  print('stroke :', self.y_predicted_myData[0])

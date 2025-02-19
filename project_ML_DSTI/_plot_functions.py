#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import plotly.express as px

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# ---------------------------------------------------------------------------------------
def plot_correlationMatrix(self, columns=[], fmt='0.2f', annot_fontsize=10, filename="correlationMatrix.png", figsize=(11,9), title=None, save=True):
  """
  Plot the correlation matrix.
  Only consider columns in the array 'columns'. Take all if columns=[].
  """
  if len(columns):
    corr = self.df[columns].corr()
  else:
    corr = self.df.corr()
    

  # ----- Plot the correlation matrix :
  mask = np.triu(np.ones_like(corr, dtype=bool)) # Remove the useless upper-triangle from the plot.
  np.fill_diagonal(mask, False) # Keep plotting the diagonal.

  # Set up the matplotlib figure
  f, ax = plt.subplots(figsize=figsize)

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(230, 20, n=256, as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin =-1,
              square=True, linewidths=.5, annot=True,
              fmt=fmt, annot_kws={'size': annot_fontsize}, 
              cbar_kws={"shrink": .75, "label":"Correlation coefficient"})

  if title:
    plt.title(title, fontsize=20)
  plt.tight_layout()
  if save:
    plt.savefig("%s/figures/%s"%(self.path,filename), bbox_inches='tight')
  else:
    plt.show()
  plt.clf()

# ---------------------------------------------------------------------------------------
def plot_hist_stack(self, x, hue, multiple="stack", save=True):
  """
  """
  my_plot = sns.histplot(data=self.df, x=x, hue=hue, multiple=multiple)

  if save:
    fig = my_plot.get_figure()
    fig.savefig("%s/figures/hist_%s_%s.png"%(self.path, x, hue), bbox_inches='tight') 
  else:
    plt.show()
  plt.clf()


# ---------------------------------------------------------------------------------------
def plot_pie(self, x, text=None, save=True):
  """
  """
  plt.pie(self.df[x].value_counts(), labels=self.df[x].value_counts().index,autopct='%1.1f%%',
          explode=[0.2,0], textprops = {"fontsize":15})
  plt.title(x, fontsize=20)
  if save:
    if text:
      plt.savefig("%s/figures/pie_%s_%s.png"%(self.path, x, text), bbox_inches='tight')
    else:
      plt.savefig("%s/figures/pie_%s.png"%(self.path, x), bbox_inches='tight')
  else:
    plt.show()
  plt.clf()

# ---------------------------------------------------------------------------------------
def plot_hist_boxplot(self, x, bins=200, save=True):
  """
  """
  plt.figure(figsize=(15,5))
  plt.subplot(1,2,1)
  plt.hist(self.df[x], bins=bins, edgecolor='white', linewidth=1.2)
  plt.xlabel(x)

  plt.subplot(1,2,2)
  sns.boxplot(x=self.df[x])
  plt.xlabel(x)

  if save:
    plt.savefig("%s/figures/hist_boxplot_%s.png"%(self.path, x), bbox_inches='tight')
  else:
    plt.show()
  plt.clf()

# ---------------------------------------------------------------------------------------
def plot_boxplot_metrics(self, save=True):
  """
  """
  for model in self.results.keys():
    for smote in [1,0]:
    
      plt.figure(figsize=(6,4))
      sns.boxplot(data=self.results[model]['smote_%s'%smote]['metrics'])
      if save:
        plt.savefig("%s/figures/boxplot_%s_smote%s.png"%(self.path,model,smote), bbox_inches="tight")
      else:
        plt.show()
      plt.clf()

# ---------------------------------------------------------------------------------------
def plot_hist(self, x, bins=200, save=True):
  """
  """
  plt.figure(figsize=(15,5))
  plt.hist(self.df[x], bins=bins, edgecolor='white', linewidth=1.2)
  plt.xlabel(x, fontsize=15)
  if save:
    plt.savefig("%s/figures/hist_%s.png"%(self.path, x), bbox_inches='tight')
  else:
    plt.show()
  plt.clf()

# ---------------------------------------------------------------------------------------
def plot_var1_bmi_missing(self, var1, xlim=None, ylim=None, title_txt=None, save=True):
  """
  """
  plt.figure(figsize=(14,5))
  plt.scatter(self.df[var1], self.df.bmi, marker='+', color='0', s=150)
  plt.scatter(self.df[var1][self.df.bmi.isna()], [30]*len(self.df[var1][self.df.bmi.isna()]), marker='o', color='red', s=100)
  if xlim: plt.xlim(xlim[0], xlim[1])
  if ylim: plt.ylim(ylim[0], ylim[1])
  plt.xlabel(var1)
  plt.ylabel("BMI")
  plt.legend([": Data", ": Missing values"])

  if save:
    if title_txt: 
      filename="%s/figures/plot_%s_bmi_missing_%s.png"%(self.path, var1, title_txt)
    else:
      filename="%s/figures/plot_%s_bmi_missing.png"%(self.path, var1)
    plt.savefig(filename, bbox_inches='tight')
  else:
    plt.show()
  plt.clf()

# ---------------------------------------------------------------------------------------
def plot_ageBMIAvgGlucoseLevel_stroke01(self, text=None, save=True):
  """
  Plot the distribution of stroke=0/1 depending on the values of 
  age, BMI and average glucose level (i.e. the three numerical values).
  """

  self.df_0 = self.df[self.df.stroke==0]
  self.df_1 = self.df[self.df.stroke==1]

  plt.figure(figsize=(14,7))
  plt.subplots_adjust(hspace=0.4, wspace=0.05)

  plt.subplot(2,2,1)
  plt.scatter(self.df_0.age, self.df_0.bmi, c='#1f77b4')
  plt.scatter(self.df_1.age, self.df_1.bmi, c='#ff7f0e')
  plt.xlabel("Age")
  plt.ylabel("BMI")
  plt.legend([": stroke = 0", ": stroke = 1"])

  ax = plt.subplot(2,2,2)
  ax.yaxis.set_label_position("right")
  ax.yaxis.tick_right()
  plt.scatter(self.df_0.avg_glucose_level, self.df_0.bmi, c='#1f77b4')
  plt.scatter(self.df_1.avg_glucose_level, self.df_1.bmi, c='#ff7f0e')
  plt.xlabel("Average Glucose Level")
  plt.ylabel("BMI")

  plt.subplot(2,2,3)
  plt.scatter(self.df_0.age, self.df_0.avg_glucose_level, c='#1f77b4')
  plt.scatter(self.df_1.age, self.df_1.avg_glucose_level, c='#ff7f0e')
  plt.xlabel("Age")
  plt.ylabel("Average Glucose Level")

  if save:
    if text:
      filename="%s/figures/plot_ageBMIAvgGlucoseLevel_stroke01_%s.png"%(self.path, text)
    else:
      filename="%s/figures/plot_ageBMIAvgGlucoseLevel_stroke01.png"%(self.path)
    plt.savefig(filename, bbox_inches='tight')
  else:
    plt.show()
  plt.clf()

# ---------------------------------------------------------------------------------------
def plot_metrics_all(self, filename="plot_metrics_all.png", save=True):
  """
  Plot the evolution of the metrics from without-SMOTE to with-SMOTE
  """

  plt.figure(figsize=(14, 7))
  plt.subplots_adjust(hspace=0.4, wspace=0.2)

  k=1
  for var in self.results[list(self.results.keys())[0]]['smote_0']['metrics'].columns:
    plt.subplot(2,2,k)
    plt.ylim([0,1])
    plt.xticks([0,1], ['no SMOTE', 'SMOTE'], ha='center')
  
    for model in self.results.keys():
     if model != 'DecisionTreeClassifier':
      recall = []
      for smote in [0, 1]:
        df = self.results[model]['smote_%d'%smote]['metrics'].mean()
        recall.append(self.results[model]['smote_%d'%smote]['metrics'][var].mean())
      plt.plot(recall, '+-')
      plt.title(var)
    
    k=k+1

  plt.legend(self.results.keys())

  if save:
    filename="%s/figures/%s"%(self.path, filename)
    plt.savefig(filename, bbox_inches='tight')
  else:
    plt.show()
  plt.clf()



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

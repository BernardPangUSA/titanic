# Using the Titanic passengers dataset to establish a model for predicting whether a given passenger would have survived the sinking of the Titanic.

#%% read data from the csv file into a data frame

import pandas as pd
import numpy as np
data = pd.read_csv("data.csv")
data

#%% replace the question marks in the age and fare columns with the numpy NaN value

data.replace("?", np.nan, inplace=True)

#%% update the column's data type

data = data.astype({"age": np.float64, "fare": np.float64})


#%% exploratory data analysis

import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=4, figsize=(30,5))
sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs[0])
sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs[1])
sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs[2])
sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs[3])

#%% convert gender to numeric values for correlation calculation

# all the variables used need to be numeric for the correlation calculation and currently gender is stored as a string. To convert those string values to integers

data.replace({'male': 1, 'female': 0}, inplace=True)

#%% analyze the correlation between all the input variables to identify the features that would be the best inputs to a machine learning model. The closer a value is to 1, the higher the correlation between the value and the result.

data.corr().abs()[["survived"]]


# %%


#%%

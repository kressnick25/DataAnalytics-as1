import pandas as pd
import numpy as np
import math
import pydot
from io import BytesIO
from sklearn.tree import export_graphviz
df = pd.read_csv('./HouseholderAtRisk.csv')
def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i] + ': ' + str(importances[i]))
        
def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = BytesIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file

# Drop ID, Weighting, Race, Gender, Education
df.drop(['ID', 'Race', 'Gender', 'Education'], axis=1, inplace=True)

### Age Column
# Age less than 1 is invalid
# Impute the invalid values and missing values with mean
# because ...
mask = df['Age'] < 1
df.loc[mask, 'Age'] = np.nan
df['Age'].fillna(df['Age'].mean(), inplace=True)

### WorkClass column
# Remove spaces
for uniq in df['WorkClass'].unique():
    if isinstance(uniq, str):
        mask = df['WorkClass'] == uniq
        df.loc[mask, 'WorkClass'] = uniq[1:]

mask = df['WorkClass'] == '?'
df.loc[mask, 'WorkClass'] = np.nan
df['WorkClass'].fillna('Private', inplace=True)

### Weighting column
df['Weighting'].fillna(df['Weighting'].mean(), inplace=True)

### NumYearsEducation column
df['NumYearsEducation'].fillna(df['NumYearsEducation'].mean(), inplace=True)

### MaritalStatus column
# Remove spaces
for uniq in df['MaritalStatus'].unique():
    if isinstance(uniq, str):
        mask = df['MaritalStatus'] == uniq
        df.loc[mask, 'MaritalStatus'] = uniq[1:]

df['MaritalStatus'].fillna('Married-civ-spouse', inplace=True)

### Occupation column
for uniq in df['Occupation'].unique():
    if isinstance(uniq, str):
        mask = df['Occupation'] == uniq
        df.loc[mask, 'Occupation'] = uniq[1:]

mask = df['Occupation'] == '?'
df.loc[mask, 'Occupation'] = np.nan
df['Occupation'].fillna('Prof-specialty', inplace=True)

### Relationship column
# Remove spaces
for uniq in df['Relationship'].unique():
    if isinstance(uniq, str):
        mask = df['Relationship'] == uniq
        df.loc[mask, 'Relationship'] = uniq[1:]

df['Relationship'].fillna('Husband', inplace=True)

### CapitalLoss column
# Impute missing values with 0 which is the median
# because the data has great outliers (Skewed to left)
df['CapitalLoss'].fillna(0, inplace=True)

### CapitalGain column
# Impute missing values with 0
df['CapitalGain'].fillna(0, inplace=True)

### CapitalAvg column
# Impute with 0
df['CapitalAvg'].fillna(0, inplace=True)

### NumWorkingHoursPerWeek column
# Impute with mean of 40
df['NumWorkingHoursPerWeek'].fillna(df['NumWorkingHoursPerWeek'].mean(), inplace=True)

### Sex column
# Impute with 0 which is the mode
df['Sex'].fillna(0, inplace=True)

### Country column
# Remove spaces 
for uniq in df['Country'].unique():
    if isinstance(uniq, str):
        mask = df['Country'] == uniq
        df.loc[mask, 'Country'] = uniq[1:]

mask = df['Country'] == '?'
df.loc[mask, 'Country'] = 'United-States'
mask = df['Country'] == 'USA'
df.loc[mask, 'Country'] = 'United-States'
mask = df['Country'] == 'US'
df.loc[mask, 'Country'] = 'United-States'
mask = df['Country'] == 'Hong'
df.loc[mask, 'Country'] = 'Hong Kong'
mask = df['Country'] == 'South'
df.loc[mask, 'Country'] = 'United-States'
df['Country'].fillna('United-States', inplace=True)

### Data types
# format Sex to binary
data_type_map = {1.0: 1, 0.0: 0}
df['Sex'] = df['Sex'].map(data_type_map)
# format Age to int
df['Age'] = df['Age'].astype(int)
# # format NumYearsEducation to int
df['NumYearsEducation'] = df['NumYearsEducation'].astype(int)
# format Weighting to int
df['Weighting'] = df['Weighting'].astype(int)
# # format AtRisk to binary
data_type_map = {'High': 1, 'Low': 0}
df['AtRisk'] = df['AtRisk'].map(data_type_map)


### One-Hot Encoding
df = pd.get_dummies(df)

columns_to_transform = ['Age', 'Weighting','CapitalGain', 'CapitalAvg']
df_log = df.copy()
for col in columns_to_transform:
    df_log[col] = df_log[col].apply(lambda x: x+1)
    df_log[col] = df_log[col].apply(np.log)

import seaborn as sns
import matplotlib.pyplot as plt
f, axes = plt.subplots(2,2, figsize=(10,10), sharex=False)
# gift avg plots
sns.distplot(df_log['Age'].dropna(), hist=False, ax=axes[0,0])
sns.distplot(df_log['Weighting'].dropna(), hist=False, ax=axes[0,1])
sns.distplot(df_log['CapitalGain'].dropna(), hist=False, ax=axes[1,0])
sns.distplot(df_log['CapitalAvg'].dropna(), hist=False, ax=axes[1,1])
# sns.distplot(df['NumYearsEducation'].dropna(), hist=False, ax=axes[1,0])
# sns.distplot(df['CapitalLoss'].dropna(), hist=False, ax=axes[1,1])

# # gift cnt plots

# sns.distplot(df['NumWorkingHoursPerWeek'].dropna(), hist=False, ax=axes[1,2])

plt.show()

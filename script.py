import pandas as pd
import numpy as np
import math
df = pd.read_csv('Householder.csv')
# Drop ID, Weighting, Race, Gender, Education
df.drop(['ID', 'Race', 'Gender', 'Education', 'CapitalLoss'], axis=1, inplace=True)

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

# Impute invalid and nan with "Private"
# because WorkClass column is categorical and only mode is the valid option
mask = df['WorkClass'] == '?'
df.loc[mask, 'WorkClass'] = np.nan
df['WorkClass'].fillna('Private', inplace=True)

### MaritalStatus column
# Remove spaces
for uniq in df['MaritalStatus'].unique():
    if isinstance(uniq, str):
        mask = df['MaritalStatus'] == uniq
        df.loc[mask, 'MaritalStatus'] = uniq[1:]

### Occupation column
# Remove spaces
for uniq in df['Occupation'].unique():
    if isinstance(uniq, str):
        mask = df['Occupation'] == uniq
        df.loc[mask, 'Occupation'] = uniq[1:]

# Invalid value is '?'
# Impute the invalid values and missing values with "Prof-speciality"
# because Occupation column is categorical and only mode is the valid option
mask = df['Occupation'] == '?'
df.loc[mask, 'Occupation'] = np.nan
df['Occupation'].fillna('Prof-specialty', inplace=True)

### Relationship column
# Remove spaces
for uniq in df['Relationship'].unique():
    if isinstance(uniq, str):
        mask = df['Relationship'] == uniq
        df.loc[mask, 'Relationship'] = uniq[1:]

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
df['NumWorkingHoursPerWeek'].fillna(40, inplace=True)

### Sex column
# Impute with 0 which is the mode
df['Sex'].fillna(0, inplace=True)

### Country column
# 
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
df['Country'].fillna('United-States', inplace=True)



#An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.

# FEATURES:

# AGE -continuous.
# workclass -Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt -continuous.
# education - Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num - continuous.
# marital-status -Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation -Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship -Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race -White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex -Female, Male.
# capital-gain -continuous.
# capital-loss -continuous.
# hours-per-week -continuous.
# native-country -United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# class - >50K, <=50K

# STEP I: IMPORTING LIBRARIES

# LIBRARIES:

# Library pandas will be required to work with data in tabular representation.
# Library numpy will be required to round the data in the correlation matrix.
# Library matplotlib, seaborn required for data visualization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#STEP II: DATA DESCRIPTION AND DATA CLEANING

#In this block, cleaning part will be carried out, data types, missing values, duplicates.

# i: Import Data

url = 'https://github.com/shiv-2025/Adult-Census-Income-Prediction/blob/main/adult.csv?raw=true'
df_raw = pd.read_csv(url)

# i: Data Types

df_raw.dtypes

#iii. Missing values
#look for null values in dataframe
df_raw.isnull().sum()

df = df_raw.drop(['education'], axis=1)

#dividing data into independent and dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#rename columns for convinience
X.columns = ['age', 'workclass', 'fnlwgt', 'education_num', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
       'hours_per_week', 'country']

#extract categorical and numerical columns
categorical_feature = [feature for feature in X.columns if X[feature].dtypes == 'O']
numerical_feature = [feature for feature in X.columns if X[feature].dtypes != 'O']

#trim whitespaces from this df
X[categorical_feature] = X[categorical_feature].apply(lambda x: x.str.strip())

#replace '?' with None values to find out null values
X = X.replace({'?': None})
X.isnull().sum()

#fill null values with median
X.fillna(X.mode().iloc[0], inplace=True)
X.isnull().sum()

# CONCLUSION:Now our Data is Clean We can do Further Analysis.

#STEP III: EXPLORATORY DATA ANALYSIS

#distribution of num values
for i in X[numerical_feature]:
    plt.hist(X[i])
    plt.title(i)
    plt.show()
    
X[numerical_feature].corr()

sns.heatmap(X.corr(), linewidths=0.1, cmap="YlGnBu")

#One hot encoding of categorical variables using get_dummies function
X_cat = pd.concat([pd.get_dummies(X[categorical_feature], drop_first=True), X[numerical_feature]], axis=1)

#Saving X & Y dataframes as csv
X_cat.to_csv('X_cat.csv', index=False)
y.to_csv('y.csv', index=False)
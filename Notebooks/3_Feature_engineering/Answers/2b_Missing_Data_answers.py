# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Loading libraries and Classes

# COMMAND ----------

# Import pandas, numpy, seaborn and matplotlib libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Import train_test_split to separate train and test set
from sklearn.model_selection import train_test_split

# Import MissingIndicator and SimpleImputer from impute module
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer

# COMMAND ----------

# Set the parameters and the style for plotting
params = {'figure.figsize':(12,8),
         'axes.labelsize':13,
         'axes.titlesize':16,
         'xtick.labelsize':11,
         'ytick.labelsize':11
         }
plt.rcParams.update(params)
sns.set_style("whitegrid")

# COMMAND ----------

# MAGIC %md
# MAGIC We'll be using Titanic dataset to explore the missing data in this notebook.

# COMMAND ----------

# Load the dataset 'Data/titanic_data.csv' and store it in variable data
data = pd.read_csv('../Data/titanic_data.csv')
# Get the first 5 rows
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. First look at the missing values
# MAGIC 
# MAGIC We can use Pandas chained `isnull().sum()` function to detect missing values.

# COMMAND ----------

# Get the total number of missing values using
data.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that three columns contain missing values: 'Age', 'Cabin' and 'Embarked'. If we want to compute the proportion of missing values, we can use `.mean()` function and plot the proportion using a barplot.

# COMMAND ----------

# Compute the proportion of missing values
percentage = data.isnull().mean()*100
percentage

# COMMAND ----------

# Plot barchart
percentage.plot(kind='bar');

# COMMAND ----------

# MAGIC %md
# MAGIC If we want to visualize the location of missing values, we can use seaborn's `heatmap` that tells us where the missing values occur. We set parameter `cbar = False` as the color bar does not need to be drawn.
# MAGIC 
# MAGIC Such a visualization has a benefit which people usually do not realize: Imagine that you just produce sums or in other words amounts of missing values in the dataset. Remember that descriptive statistics might reveal less than what visualisation does. This is also true for missing values. You might be able to spot, for example, **that missing values in two columns have a similar or the same pattern**. 

# COMMAND ----------

# Visualize only those three columns that contain missing values
data_copy = data[['Age','Cabin','Embarked']]
sns.heatmap(data_copy.isnull(), cbar = False);

# COMMAND ----------

# MAGIC %md
# MAGIC For even better visualization of missing values, we can again use the dedicated library [missingno](https://github.com/ResidentMario/missingno).

# COMMAND ----------

import missingno as msno
fig, ax = plt.subplots(figsize=(10,6))
msno.heatmap(data, ax=ax);

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10,6))
msno.dendrogram(data, ax=ax);

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Concepts of missing values
# MAGIC 
# MAGIC According to Rubin's theory \\(^{1}\\), every data point has some probability of being missing in the dataset. The process that governs these probabilities is called **the missing data mechanism**. 
# MAGIC 
# MAGIC ## 2.1 MNAR: Missing data Not At Random
# MAGIC 
# MAGIC MNAR means that the probability of being missing varies for reasons that are unknown to us. Let's look at the columns 'Age' and 'Cabin'. We found out that the column 'Cabin' contains approximately 77% missing values, the column 'Age' almost 20% missing values. 
# MAGIC 
# MAGIC The age or cabin could not be established for people who did not survive that night. We assume that survivors were asked for such information. But can we infer this when we look at the data? In this case, we expect that observations of people who did not survive should have more missing values. Let's find out.
# MAGIC 
# MAGIC *Note: Below is a cool functionality of pandas. The method is called query and allows you to really simply subset your data. Of course you could also solve with the traditional functionality which you already learned, I just wanted to make use of the opportunity.*
# MAGIC 
# MAGIC ### 2.1.1 Diagnosing Missing Data

# COMMAND ----------

# Filter the dataset based on people who survived
survived = data.query('Survived == 1')
survived

# COMMAND ----------

# Print the percentage of missing values in the column 'Cabin' for people who survived
print('The percentage of missing values: {0:.1f} %'.format(survived['Cabin'].isna().mean()*100))

# COMMAND ----------

# Filter the dataset based on people who did not survived
not_survived = data.query('Survived == 0')
not_survived

# COMMAND ----------

# Print the percentage of missing values in the column 'Cabin' for people who did not survive
print('The percentage of missing values: {0:.1f} %'.format(not_survived['Cabin'].isna().mean()*100))

# COMMAND ----------

# MAGIC %md
# MAGIC The output we obtained is the same as we expected. There are more missing values (approximately 87.6%) for people who did not survive compared to those that did (60.2 %).

# COMMAND ----------

# TASK 1 >>>> Now it's your turn to explore the column 'Age' in the same way 
#             and think about whether the values are missing at random or not

print(survived['Age'].isna().mean()*100)
print(not_survived['Age'].isna().mean()*100)

# COMMAND ----------

# MAGIC %md
# MAGIC In the previous notebook we've filled in missing values using Pandas `fillna()` method. We can specify a scalar value method such as backward fill ('bfill'), or forward fill ('ffill'), or statistic such as mean, median, or mode of the particular column within this method. If we want to replace missing data with 'bfill' method or 'ffill' method and the previous or the next value is not present, the missing values remain present after the imputation. Also, be careful when filling in missing values with the mean if your data has outliers since the mean is affected by them.
# MAGIC 
# MAGIC This approach of filling missing values should be sufficient if you will use the dataset for simple analysis. However, remember what we discussed: As soon as we want to build a robust pipeline, for example for Machine Learning, we need to be able to save the state. This means that the Pandas functionality may not be the best one. We would need to be manually saving the state of *"mean which should be imputed"* somewhere. 
# MAGIC 
# MAGIC Luckily, scikit-learn offers a handy alternative in forms of **missing indicator** and **simple imputer**. Both of these are saving the state so that we can easily make those part of our robust pipeline. Let's now take a look at these two.
# MAGIC 
# MAGIC 
# MAGIC ------
# MAGIC 
# MAGIC **Simple Imputer and Missing Indicator**
# MAGIC 
# MAGIC `scikit learn` offers transformers for univariate and multivariate imputation of missing values. You can read more in the [documentation](https://scikit-learn.org/stable/modules/impute.html). Now we demonstrate the usability of the `SimpleImputer()` class from the impute module. You can specify several parameters, such as the placeholder (`np.nan`) for missing values, the imputation strategy, or the value used to replace missing values. Find more [here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **Before we impute the missing values**, it is useful to mark missing values to preserve the information about which values had been missing. We can use `MissingIndicator`, which transforms the dataset into binary variables indicating the presence of missing values (these binary variables will be added to the original training set). See the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html#sklearn.impute.MissingIndicator) for `MissingIndicator`.
# MAGIC 
# MAGIC In essence, the idea behind `MissingIndicator` is that we **preserve extra information** for our model which is if the value was missing. We are hoping that the model might pick up a pattern herein which we missed.
# MAGIC 
# MAGIC Let's split our data into training and testing set, mark missing values, and fill in those using `SimpleImputer`.

# COMMAND ----------

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data[['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                                                        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']], 
                                                    data['Survived'],
                                                    test_size = 0.3,
                                                    random_state = 42) 
# Get the shape 
X_train.shape, X_test.shape

# COMMAND ----------

# Get the number of missing values
X_train.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.2 Missing Indicator
# MAGIC We'll use `MissingIndicator` to mark missing values by setting the parameter `features = 'missing-only'` (by default). If we want to create binary variables for all features, we set the parameter to `features = 'all'`.

# COMMAND ----------

# Create MissingIndicator object
missing_indicator = MissingIndicator(features = 'missing-only')

# COMMAND ----------

# Fit X_train with missing_indicator 
missing_indicator.fit(X_train)

# COMMAND ----------

# Get the features with missing values
missing_indicator.features_

# COMMAND ----------

# MAGIC %md
# MAGIC With `.features_` attribute we get feature names or the indices of features containing missing values. 

# COMMAND ----------

# Get the features names
X_train.columns[missing_indicator.features_]

# COMMAND ----------

# MAGIC %md
# MAGIC Since the transformation of `X_train` returns a boolean, we create a new variable to store the output. After that, we concatenate it to the original `X_train`.

# COMMAND ----------

# Transform X_train using missing_indicator and store the output to X_missing
X_train_missing = missing_indicator.transform(X_train)

# COMMAND ----------

# Display the output
X_train_missing

# COMMAND ----------

# MAGIC %md
# MAGIC Now we concatenate the `X_train_missing` boolean variables to the original `X_train`. To distinguish which boolean variable belongs to the original feature in `X_train`, we create new names (otherwise, boolean variables names will be labeled as 0,1 and 2). `X_train_missing` array needs to be converted using `pd.DataFrame`, since only Series and DataFrame objects are valid within the `concat()` method.

# COMMAND ----------

# Create new column names for boolean variables
# Create an empty list to store the new names
indicator_name = []
# Iterate over the features with missing values
for column in X_train.columns[missing_indicator.features_]:
    column_name = column + '_Missing'
    # Append new names to the indicator_name list
    indicator_name.append(column_name)

# COMMAND ----------

# Concatenate the original X_train and X_train_missing along the columns
# Reset the index in X_train and convert X_train_missing to a Pandas DataFrame 
# Assign new column names stored in indicator_name to the columns parameter
X_train = pd.concat([X_train.reset_index(), pd.DataFrame(X_train_missing, columns = indicator_name)], axis = 1)
X_train

# COMMAND ----------

# TASK 2 >>>>> Repeat the process for X_test 
# Transform X_test data using missing_indicator and store it to the variable X_test_missing

X_test_missing = missing_indicator.transform(X_test)

# COMMAND ----------

# TASK 2 >>>>> Concatenate the original X_test and X_test_missing along the columns in the same way as we did for X_train
# Assign it to the original X_test

X_test = pd.concat([X_test.reset_index(), pd.DataFrame(X_test_missing, columns = indicator_name)], axis = 1)

# Display X_test to see the result
X_test

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.3 Simple Imputer
# MAGIC Now we'll impute the missing values of the column 'Age' using `SimpleImputer`. We specify the placeholder for missing values (`np.nan`) and `strategy = 'mean'` (this strategy is by default, so it is okay if you don't explicitly specify it within the class).

# COMMAND ----------

# Create SimpleImputer object for imputing missing values with mean strategy
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# COMMAND ----------

# Fit column 'Age' in X_train
imputer.fit(X_train[['Age']])

# COMMAND ----------

# To see the mean value that will be used for imputing we can use .statistics_ attribute
imputer.statistics_

# COMMAND ----------

# Transform column 'Age' in X_train
X_train['Age'] = imputer.transform(X_train[['Age']])

# COMMAND ----------

# Get the total number of missing values in column 'Age' to see whether these values have been replaced
X_train['Age'].isnull().sum()

# COMMAND ----------

# TASK 3 >>>>> Repeat the imputing also for column 'Age' in X_test data
X_test['Age'] = imputer.transform(X_test[['Age']])

# # Get the total number of missing values in column Age to see whether these values have been replaced
X_train['Age'].isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 MCAR: Missing data Completely At Random 
# MAGIC 
# MAGIC When data is missing completely at random, the probability of being missing is the same for all observations in the dataset, i.e. the cause of the missing data is unrelated to the data.
# MAGIC 
# MAGIC Let's take as an example the column 'Embarked' and its missing values.

# COMMAND ----------

# Get the rows where the values in the 'Embarked' column are missing
data[data['Embarked'].isnull()]

# COMMAND ----------

# MAGIC %md
# MAGIC Mrs. Stone was traveling in the first class with her maid Miss. Amelie Icard. They occupied the same Cabin B28, but the data on the port of embarkation is missing. But we cannot tell if the 'Embarked' variable depends on any other variable. We can also see that these women have survived, so we assume that they were asked for that information. It could happen that this information was lost when the dataset was created. The probability of losing this information is the same for every person on the Titanic. However, this would probably be impossible to prove. 
# MAGIC 
# MAGIC For curiosity: You can find out more information about Mrs. Stone and her maid [here](https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html). There is also information about the port of embarkation in this article.  
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC We can impute missing values also in the case of categorical variables that have values stored as strings. Let's impute the missing values of the 'Embarked' column in the `X_train` data. We set `strategy = constant` that allows us to specify the `fill_value` used to replace missing values. This can be used with strings or numeric data as well. The second option for strategy is `most_frequent` when the missing values will be replaced using the most frequent column value.

# COMMAND ----------

# Create SimpleImputer object and store it to variable imputer_cat
imputer_cat = SimpleImputer(missing_values = np.nan, strategy = 'constant',fill_value = 'S')

# COMMAND ----------

# Fit column 'Embarked' in X_train
imputer_cat.fit(X_train[['Embarked']])

# COMMAND ----------

# Transform column 'Embarked' in X_train
X_train['Embarked'] = imputer_cat.transform(X_train[['Embarked']])

# COMMAND ----------

# Get the total number of missing values in column 'Embarked' to see whether these values have been replaced
X_train['Embarked'].isnull().sum()

# COMMAND ----------

# TASK 4 >>>>> Repeat the imputing also for the column 'Embarked' in X_test data

X_test['Embarked'] = imputer_cat.transform(X_test[['Embarked']])

# Get the total number of missing values in column Embarked to see whether these values have been replaced
X_test['Embarked'].isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 MAR: Missing At Random
# MAGIC 
# MAGIC We can say that the data is missing at random if the probability of being missing is the same only within groups defined by the observed data. An example of this case is when we take a sample from a population. The probability to be included depends on some known property. For example, when placed on a soft surface, a weighing scale may produce more missing values than when placed on a hard surface.

# COMMAND ----------

# MAGIC %md
# MAGIC ### TASK
# MAGIC 
# MAGIC In this task you will be using the Avocado dataset. You will impute numeric missing values in the column 'Small Bags' with the median value using `SimpleImputer`. The second task is to impute missing values in the column 'Region' with the most frequent string value of this column also using `SimpleImputer`.

# COMMAND ----------

# Load the dataset 'Data/avocado_missing.csv' and store it to variable avocado
avocado = pd.read_csv('../Data/avocado_missing.csv')
# Print the first 5 rows
avocado.head()

# COMMAND ----------

# TASK >>>> Print the total number of missing values

avocado.isnull().sum()

# COMMAND ----------

# TASK >>>> Create SimpleImputer object and store it in variable imputer_median
# Specify that you want to impute the median value

imputer_median = SimpleImputer(missing_values = np.nan, strategy = 'median')

# COMMAND ----------

# TASK >>>> Fit column 'Small Bags' using imputer_median 

imputer_median.fit(avocado[['Small Bags']])

# COMMAND ----------

# Print the median value that will be used to replace the missing values
imputer_median.statistics_

# COMMAND ----------

# TASK >>>> Transform the column 'Small Bags' using imputer_median
# Assign the transformation to avocado['Small Bags']

avocado['Small Bags'] = imputer_median.transform(avocado[['Small Bags']])

# COMMAND ----------

# TASK >>>> Create a SimpleImputer object and store it in the variable imputer_freq
# Specify that you want to impute the most frequent value

imputer_freq = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

# COMMAND ----------

# TASK >>>> Fit the column 'region' using imputer_freq

imputer_freq.fit(avocado[['region']])

# COMMAND ----------

# Print the most frequent value that will be used to replacing missing values
imputer_freq.statistics_

# COMMAND ----------

# TASK >>>> Transform column 'region' using imputer_freq
# Assign the transformation to avocado['region']

avocado['region'] = imputer_freq.transform(avocado[['region']])

# COMMAND ----------

# Print the total number of missing values to see whether the missing values have been replaced
avocado.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Read only - Column Transformer
# MAGIC 
# MAGIC Commonly, preparing data for machine learning models often involves several transformations such as imputing missing values, scaling numerical values, or encoding categorical features applied for particular columns. `scikit learn` offers the [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) class that is used to apply different transformers to columns. This column transformer can be chained with Pipelines along with machine learning model. You can read more about `ColumnTransformer` [here](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix
# MAGIC 
# MAGIC \\(^{1}\\) Inference and missing data, DONALD B. RUBIN, Biometrika, Volume 63, Issue 3, December 1976, Pages 581–592
# MAGIC 
# MAGIC Data source: 
# MAGIC 
# MAGIC Titanic dataset: https://www.kaggle.com/hesh97/titanicdataset-traincsv
# MAGIC 
# MAGIC Data license: CC0: Public Domain
# MAGIC 
# MAGIC Avocado dataset: https://www.kaggle.com/neuromusic/avocado-prices
# MAGIC 
# MAGIC Data license: Database: Open Database
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)

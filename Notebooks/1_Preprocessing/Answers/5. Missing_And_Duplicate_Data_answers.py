# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Missing values
# MAGIC 
# MAGIC In the real world, the data are rarely clean and homogenous and can have missing values for several reasons: data was lost or corrupted during the transmission from the database, human error, programming error. Whether the missing data will be removed, replaced or filled depends on the type of missing data.
# MAGIC 
# MAGIC `Pandas` uses the floating point value `NaN` (Not a Number) to represent missing data in both numeric as well as in non-numeric datatypes. The built-in Python `None` value is also treated as NA in object arrays.
# MAGIC 
# MAGIC There are several functions for detecting, removing, replacing and imputing null values in Pandas DataFrame.

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look how the missing data look like in the DataFrame.

# COMMAND ----------

# Run this code
our_series = pd.Series([25, 2.5, 150, np.nan, 1.5, 'Python', 147])
print(our_series)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Detecting missing data

# COMMAND ----------

# Load the Titanic dataset
data = pd.read_csv('../Data/titanic.csv')
data.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC `isnull().values.any()`
# MAGIC 
# MAGIC - used if we only want to know if there are any missing values in the dataset

# COMMAND ----------

# Check whether there are any missing values
data.isnull().values.any()

# COMMAND ----------

# MAGIC %md
# MAGIC `isnull()`
# MAGIC - it is used to detect missing values for an array-like object
# MAGIC - returns a boolean same-sized object indicating if the values are missing
# MAGIC 
# MAGIC - it is an alias of `isna()`

# COMMAND ----------

# Apply isnull() on the dataset 'data'
data.isnull()

# COMMAND ----------

# MAGIC %md
# MAGIC `notnull()`
# MAGIC 
# MAGIC - it is used to detect existing (non-missing) values
# MAGIC - it is an alias of `notna()`

# COMMAND ----------

# TASK 1 >>>> Check non-missing values in the dataset using .notnull()

data.notnull()

# COMMAND ----------

# MAGIC %md
# MAGIC `isnull().sum()`
# MAGIC 
# MAGIC - we can use function chaining to check the total number of missing values for each column in the DataFrame

# COMMAND ----------

# Count the total number of missing values in the column using .sum()
data.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, there are 177 missing values in the column Age, then 687 missing values in the column Cabin and 2 missing values in the Embarked column.

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Basic visualization of missing data

# COMMAND ----------

# Run this code
plt.style.use('default')
missing_values = data.isnull().sum() / len(data) * 100
plt.xticks(np.arange(len(missing_values)), missing_values.index,rotation='vertical')
plt.ylabel('Percentage of missing values')
ax = plt.bar(np.arange(len(missing_values)), missing_values, color = 'skyblue');

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Removing missing data

# COMMAND ----------

# MAGIC %md
# MAGIC In some cases, it is appropriate just drop the rows with missing data, in other cases replacing missing data would be better options. 
# MAGIC 
# MAGIC `dropna()` function $^{1}$
# MAGIC 
# MAGIC - to remove rows or columns from the DataFrame which contain missing values
# MAGIC - by default drops any row that contain a missing value
# MAGIC 
# MAGIC Arguments:
# MAGIC 
# MAGIC `axis = 0` to drop rows
# MAGIC 
# MAGIC `axis = 1` to drop columns
# MAGIC 
# MAGIC `how = 'all'` to drop if all the values are missing
# MAGIC 
# MAGIC `how = 'any'` to drop if any missing value is present
# MAGIC 
# MAGIC `tresh = ` treshold for missing values
# MAGIC 
# MAGIC `subset = ['column']` to remove rows in which values are missing or selected column or columns
# MAGIC 
# MAGIC **If we want to make changes in the original dataset** (for example remove a particular column), we have to specify `inplace = True` within the method. Otherwise the copy of the dataset will be returned and the change will not be performed in the original dataset. 

# COMMAND ----------

# Print rows with missing data in the column 'Embarked'

missing_embarked = data[data.Embarked.isnull()]
print(missing_embarked)

# COMMAND ----------

# Drop missing values in the column 'Embarked' 
# Specify this column using subset
# Set inplace = True

data.dropna(subset = ['Embarked'], inplace = True)

# COMMAND ----------

# Check whether the rows with missing values have been removed
data.Embarked.isna().sum()

# COMMAND ----------

# Make a copy of the DataFrame
data_copy = data.copy()

# COMMAND ----------

# Drop those rows that contain any missing values
# Set inplace = True

data_copy.dropna(how = 'any', inplace = True)

# COMMAND ----------

# Check whether the rows have been removed correctly

data_copy.isna().sum()

# COMMAND ----------

# Run this code
dict = {'product': ['apple', np.nan,'cucumber','bread','milk', 'butter', 'sugar'],
        'product_code': [154,153,225,np.nan,56,15, np.nan],
        'price': [0.89, 1.50, 0.65, 1.20, 0.85, np.nan, 1.20],
        'expiration_date': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        }

df = pd.DataFrame (dict, columns = ['product','product_code','price','expiration_date'])
print(df)

# COMMAND ----------

# Drop the last column that contain only missing values
# Set inplace = True

df.dropna(how = 'all', axis = 1, inplace = True)

# COMMAND ----------

# Display the DataFrame to check the change

df

# COMMAND ----------

# Run this code

df_copy = df.copy()
print(df_copy)

# COMMAND ----------

# TASK 2 >>>> Drop rows from df_copy that contain any missing values 
#             Set inplace = True

df_copy.dropna(how = 'any',inplace = True)
print(df_copy)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Filling in missing data
# MAGIC 
# MAGIC `fillna()` method
# MAGIC 
# MAGIC - this method fill in missing data (can be used on a particular column as well)
# MAGIC 
# MAGIC Arguments:
# MAGIC 
# MAGIC - we can specify **value** (any number or summary statistics such as mean or median) 
# MAGIC 
# MAGIC - we can use **interpolation method**: 
# MAGIC 
# MAGIC `ffill` : uses previous valid values to fill gap
# MAGIC 
# MAGIC `bfill` : uses next valid value to fill gap
# MAGIC 
# MAGIC `limit` : for ffill and bfill - maximum number of consecutive periods to fill
# MAGIC 
# MAGIC `axis` : axis to fill on, default axis = 0 
# MAGIC 
# MAGIC `inplace = True`

# COMMAND ----------

# Fill in missing value in 'price' column with value 0
# Set inplace = True

df.price.fillna(0, inplace = True)
print(df)

# COMMAND ----------

# Run this code
dictionary = {'column_a': [15, 16, 82, 25],
              'column_b': [np.nan, np.nan, 54, 8],
              'column_c': [np.nan, 15, 15, 25],
              'column_d': [85, 90, np.nan, np.nan]
        }

dataframe_1 = pd.DataFrame (dictionary, columns = ['column_a','column_b','column_c','column_d'])
print(dataframe_1)

# COMMAND ----------

# TASK 3 >>>> Fill in missing value in column 'column_c' of dataframe_1 with value 10 
#             Set inplace = True

dataframe_1.column_c.fillna(10, inplace = True)
print(dataframe_1)

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. More Complex Methods
# MAGIC 
# MAGIC We will go through the theory of these more complex methods later as they relate to Machine Learning. 

# COMMAND ----------

# Run this code
dict = {'column_1': [15, 16, 82, 25],
        'column_2': [np.nan, np.nan, 54, 8],
        'column_3': [np.nan, 15, 15, 25],
        'column_4': [85, 90, np.nan, np.nan]
        }

our_df = pd.DataFrame (dict, columns = ['column_1','column_2','column_3','column_4'])
print(our_df)

# COMMAND ----------

# Fill in missing values using 'method = bfill' which stand for 'backward fill'
# Set inplace = True

our_df.fillna(axis = 0, method = 'bfill', inplace = True)
print(our_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The second option is `method = 'ffill'` which stand for forward fill.

# COMMAND ----------

# Convert the datatype of the column Age from the DataFrame 'data' to integer data type

data_copy.Age = data_copy.Age.astype('int')

# COMMAND ----------

# Fill in missing data of the column 'Age' in the DataFrame 'data' with the average age
# Set inplace = True

average_age = data_copy.Age.mean()
data_copy.Age.fillna(average_age, inplace = True)

# COMMAND ----------

# Check whether missing values have been removed from the column 'Age'

data_copy.Age.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Duplicate data

# COMMAND ----------

# Run this code
actors = [('Michone', 30, 'USA'),
            ('Bob', 28, 'New York'),
            ('Rick', 30, 'New York'),
            ('Carol', 40, 'Paris'),
            ('Daryl', 35, 'London'),
            ('Daryl', 35, 'London'),
            ('Michone', 45, 'London'),
            ('Morgan', 38, 'Sweden')
            ]
df_2 = pd.DataFrame(actors, columns=['first_name', 'age', 'city'])

# COMMAND ----------

# Find duplicated values using .duplicated() method

df_2.duplicated().sum()

# COMMAND ----------

# Remove duplicate rows
# Set inplace = True

df_2.drop_duplicates(inplace=True)
print(df_2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix
# MAGIC 
# MAGIC Data source: https://www.kaggle.com/hesh97/titanicdataset-traincsv
# MAGIC 
# MAGIC License: CC0: Public Domain
# MAGIC 
# MAGIC # References
# MAGIC 
# MAGIC 1. Pandas documentation. 2020. pandas.DataFrame.dropna. [ONLINE] Available at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html. [Accessed 14 September 2020].
# MAGIC 
# MAGIC 2. Pandas documentation. 2020. pandas.DataFrame.fillna. [ONLINE] Available at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html. [Accessed 14 September 2020].
# MAGIC 
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science) 

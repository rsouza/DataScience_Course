# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Loading libraries and data

# COMMAND ----------

# Import pandas library 
import pandas as pd

import warnings
warnings.simplefilter("ignore")

from IPython.display import Image

# Import train_test_split to separate train and test set
from sklearn.model_selection import train_test_split

# Import MinMaxScaler to scale the features
from sklearn.preprocessing import MinMaxScaler

# COMMAND ----------

# Load avocado dataset and store it in the variable dataframe
dataframe = pd.read_csv('Data/avocado.csv')
# Get the first 5 rows
dataframe.head()

# COMMAND ----------

# Print a summary of the dataframe
dataframe.info()

# COMMAND ----------

# Use numerical variables to create DataFrame data
data = dataframe.select_dtypes(include = ['float64'])

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Variable scale / Magnitude

# COMMAND ----------

# Print descriptive statistics of these variables to see variable's magnitudes
data.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, our variables have different magnitudes/scales. The minimum and maximum values of the variables are different. For example, the minimum and maximum value for the average price of avocados are 0.44 and 3.25, respectively. For small bags of avocados sold, the minimum and maximum values are 0 and 1.338459e+07, respectively.

# COMMAND ----------

# Get the range of numerical variables
for col in['AveragePrice', 'Total Volume', 'Small Hass Avocado','Large Hass Avocado', 'Extra Large Hass Avocado', 'Total Bags',
            'Small Bags', 'Large Bags', 'XLarge Bags']:
    print(col, 'range is: ', data[col].max() - data[col].min())

# COMMAND ----------

# MAGIC %md
# MAGIC The ranges of our variables are all different! 
# MAGIC 
# MAGIC # 2. Feature Scaling
# MAGIC 
# MAGIC Models such as logistic regression, linear regression – or other models that involve a matrix – are very sensitive to different scales of input variables. If we use such data for model fitting, the result might end up creating a bias. Therefore feature scaling techniques are used before model fitting.
# MAGIC 
# MAGIC As you can guess, feature scaling techniques change the scale of the variables. There are several ways how you can scale your features. In this notebook we'll demonstrate the **MinMaxScaling** technique that scales variables to their minimum and maximum values. scikit learn offers the `MinMaxScaler` class for this purpose. You can find the documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
# MAGIC 
# MAGIC The formula for min-max scaling is: 
# MAGIC 
# MAGIC **X_std = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))**
# MAGIC 
# MAGIC **X_scaled = X_std * (max - min) + min**
# MAGIC 
# MAGIC - our Scaler subtracts the minimum value from all observations in our dataset and divide it by the range of values
# MAGIC - it will transform each feature individually between 0 and 1 

# COMMAND ----------

# Let's split our dataset to training and testing set
X_train, X_test, y_train, y_test = train_test_split(data[['Total Volume', 'Small Hass Avocado','Large Hass Avocado', 
                                                          'Extra Large Hass Avocado', 'Total Bags', 
                                                          'Small Bags', 'Large Bags', 'XLarge Bags']],
                                                    data['AveragePrice'],
                                                    test_size = 0.3,
                                                    random_state = 42)
# Get the shape of X_train and X_test
X_train.shape, X_test.shape

# COMMAND ----------

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# COMMAND ----------

# Fit X_train with scaler: this computes and saves the minimum and maximum values 
scaler.fit(X_train)

# COMMAND ----------

# We can access the maximum values using the .data_max attribute
scaler.data_max_

# COMMAND ----------

# We can access the minimum values using .data_min attribute
scaler.data_min_

# COMMAND ----------

# Transform X_train and X_test with scaler and store it in the variables X_train_scaled and X_test_scaled
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# Let's have a look at the scaled training dataset
print('Mean: ', X_train_scaled.mean(axis=0))
print('Standard Deviation: ', X_train_scaled.std(axis=0))
print('Minimum value: ', X_train_scaled.min(axis=0))
print('Maximum value: ', X_train_scaled.max(axis=0))

# COMMAND ----------

# MAGIC %md
# MAGIC After this rescaling, all of the features have a range from 0 to 1.

# COMMAND ----------

# Let's have a look at the scaled testing dataset
print('Mean: ', X_test_scaled.mean(axis=0))
print('Standard Deviation: ', X_test_scaled.std(axis=0))
print('Minimum value: ', X_test_scaled.min(axis=0))
print('Maximum value: ', X_test_scaled.max(axis=0))

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the range of the features in the test set is not exactly 0 to 1. This is because `MinMaxScaler` has only been trained on the training data `X_train`, not on `X_test`, to prevent data leakage!

# COMMAND ----------

# MAGIC %md
# MAGIC ### TASK
# MAGIC 
# MAGIC Imagine you've normalized the data using `MinMaxScaler` and delivered your work to the Senior Data scientist. He/she proposed you to scale the data using different scaling technique. The technique should transform the data such that its distribution will have a mean value of 0 and a standard deviation of 1. Find the right method [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing).

# COMMAND ----------

# TASK >>>> Import the selected Scaling class

# TASK >>>> Create a scaler object

# COMMAND ----------

# Fit X_train using scaler_technique
scaler_technique.fit(X_train)

# COMMAND ----------

# Print the scaled values
scaler_technique.mean_

# COMMAND ----------

# Transform X_train using scaler_technique and store it in variable X_training_scaled
X_training_scaled = scaler_technique.transform(X_train)

# Print X_training_scaled
X_training_scaled

# COMMAND ----------

# Repeat the scaling also for X_test and store it in variable X_testing_scaled
X_testing_scaled = scaler_technique.transform(X_test)

# Print X_testing_scaled
X_testing_scaled

# COMMAND ----------

# Print mean and standard deviations of X_training_scaled
print('Mean: ', X_training_scaled.mean(axis=0))
print('Standard Deviation: ', X_training_scaled.std(axis=0))

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix
# MAGIC 
# MAGIC Data source: 
# MAGIC 
# MAGIC Avocado dataset: https://www.kaggle.com/neuromusic/avocado-prices
# MAGIC 
# MAGIC Data license: Database: Open Database
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)

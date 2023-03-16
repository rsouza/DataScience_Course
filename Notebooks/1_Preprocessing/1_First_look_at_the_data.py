# Databricks notebook source
# data source: https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Must-do Exploration
# MAGIC In this section we will be exploring the data on Cars used for Testing fuel economy (car models are from year 2010).

# COMMAND ----------

# TASK 1 >>>
# import pandas as pd
# import numpy as np
# Load the dataset 'fuel_economy_data_2010.csv' using pd.read_csv('...', index_col = 0) and store it in a variable called "data"

# COMMAND ----------

# TASK 2 >>> Display the first five rows of the DataFrame using .head() method

# COMMAND ----------

# TASK 3 >>> Display the last five rows of the DataFrame using .tail() method

# COMMAND ----------

# Display 15 random elements from the dataset using .sample() method (by setting parameter n = 15)
data.sample(n=15)

# COMMAND ----------

# Display 5 random elements from the column 'Test Veh Displacement (L)' using .sample(n = 5)
data['Test Veh Displacement (L)'].sample(n = 5)

# COMMAND ----------

# TASK 4 >>> Display the shape of the dataset using attribute .shape

# COMMAND ----------

# Display the short summary of the dataframe using .info() method by setting argument verbose = False
data.info(verbose = False)

# COMMAND ----------

# TASK 5 >>> Display a basic summary of the DataFrame using .info() method

# COMMAND ----------

# TASK 6 >>> Display columns in the dataframe using .columns attribute

# COMMAND ----------

# TASK 7 >>> Display data types of the variables using .dtypes attribute

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Exploring columns

# COMMAND ----------

# Display a subset of the dataframe's columns based on the column's data types, using .select_dtypes() method 
# Set parameter include=['float64','int64']
data.select_dtypes(include=['float64', 'int64'])

# COMMAND ----------

# TASK 8 >>> Display descriptive statistics using .describe() method

# COMMAND ----------

# Display summary statistics of all columns of the dataframe regardless of data type by setting argument include = 'all'
data.describe(include = 'all')

# COMMAND ----------

# Display summary statistics of string columns only by setting argument include=[object]
data.describe(include = [object])

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Exploring particular column
# MAGIC  

# COMMAND ----------

# Display counts of unique values of the column 'Tested Transmission Type' using .value_counts() method
data['Tested Transmission Type'].value_counts()

# COMMAND ----------

# Display relative frequencies of the unique values of the column 'Tested Transmission Type' by setting argument normalize = True
data['Tested Transmission Type'].value_counts(normalize = True)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Using boxplot

# COMMAND ----------

# Import seaborn as sns and matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Set the size of the figure
plt.figure(figsize = (9,10))

# Create a boxplot to show the distribution of datapoints of the column 'Rated Horsepower'
sns.boxplot(data['Rated Horsepower'], orient = 'v', linewidth = 2, color ='skyblue');

# COMMAND ----------

# Set the size of the figure
plt.figure(figsize = (9,10))

# TASK 9 >>> Create a boxplot to show the distribution of datapoints of the column 'Test Veh Displacement (L)' that describes engine displacement

# COMMAND ----------

# MAGIC %md
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. Source: https://github.com/zatkopatrik/authentic-data-science

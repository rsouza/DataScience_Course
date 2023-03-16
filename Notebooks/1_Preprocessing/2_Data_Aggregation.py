# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Importing data (DO NOT ALTER)

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# Load the dataset
data = pd.read_csv('Data/avocado.csv')

# COMMAND ----------

# Preview the data
data

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. 1-D aggregations on Pandas Series
# MAGIC 
# MAGIC Let's recall computing aggregations such as `sum()`, `mean()`, `median()`, `max()` and `min()` using Pandas Series. 

# COMMAND ----------

# Create Pandas Series using values: [8.45, 3.15, 1.25, 10.55, 2.40]

our_series = pd.Series([8.45, 3.15, 1.25, 10.55, 2.40])

# COMMAND ----------

# TASK 1 >>>> Print computing aggregations 
print(f'The rounded count of the values is: {our_series.sum().round()}')

# Fill in the empy brackets {} (returns error if not filled)
print(f'The average value is: {}')
print(f'The median value is: {}')
print(f'The maximum value is: {}')
print(f'The minimum value is: {}')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. 2-D aggregations on Pandas DataFrame
# MAGIC 
# MAGIC To understand the true power of `groupby()` we can take a look at what is going on under the hood.  
# MAGIC Let's say we want to compute the average price of avocados based on their type: conventional and organic. 
# MAGIC 
# MAGIC Firstly, we have to split our dataset into 2 different groups based on the type:

# COMMAND ----------

# Filter only those records that are organic type and assign it to variable filter_o
filter_o = data['type'] == 'organic'

# COMMAND ----------

# Use .loc[] on data to access all columns based on our condition filter_o and assign it to the variable data_organic
data_organic = data.loc[filter_o]
data_organic

# COMMAND ----------

# MAGIC %md
# MAGIC See that only organic type remain.

# COMMAND ----------

# TASK 2.1 >>>> Filter only those records that are of type conventional and assign it to the variable filter_c

# COMMAND ----------

# TASK 2.2 >>>> Use .loc[] on data to access all columns based on our condition filter_c and assign it to the variable data_conventional

# COMMAND ----------

# MAGIC %md
# MAGIC Now compute the average price for both types of avocados using the `.mean()` method applied to the column `AveragePrice`.

# COMMAND ----------

# Compute the average price for filtered organic avocados and assign it to the variable avg_organic
avg_organic = data_organic['AveragePrice'].mean()

# COMMAND ----------

# TASK 3 >>>> Compute the average price for filtered conventional avocados and assign it to the variable avg_conventional

# COMMAND ----------

# Print the outputs and the type of the outputs
print(avg_organic, avg_conventional)
print('\n')
print(type(avg_organic), type(avg_conventional))

# COMMAND ----------

# MAGIC %md
# MAGIC Lastly, combine these results into data structure using `Pandas`  `.DataFrame()`. Create a dictionary, where the first key name will be 'Type' and its values 'organic', 'conventional'. The second key name will be 'Average_price' and its values will be our created `avg_organic` and `avg_conventional`, respectively.

# COMMAND ----------

# Combine these results into a new DataFrame
data_output = pd.DataFrame({'Type':['organic','conventional'], 
                            'Average_price':[avg_organic, avg_conventional]})

# COMMAND ----------

# Print the resulting DataFrame
print('\nResult dataframe :\n',data_output)

# COMMAND ----------

# MAGIC %md
# MAGIC However, we can use `groupby()` to achieve the same result with only 1 line of the code!

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. 2-D aggregations on Pandas DataFrame (KEY LEARNING)

# COMMAND ----------

# MAGIC %md
# MAGIC  `groupby()` function allows us to quickly and efficiently split the data into separate groups to perform computations. When we pass the desired column or columns within `groupby()`, it will return _DataFrameGroupBy object_. We can think of it as a special view on our DataFrame. No computation will be done until we specify functions such as `mean()`, `sum()` etc. 

# COMMAND ----------

# Group the data based on the column 'year'
data.groupby('year')

# COMMAND ----------

# MAGIC %md
# MAGIC Now we compute the average price for organic and conventional avocados again but we'll make use of `groupby()`.

# COMMAND ----------

# Group the data based on Avocado type
# Compute the average price using .mean()

by_type_total = data.groupby('type')['AveragePrice'].mean()
print(by_type_total)

# COMMAND ----------

# Group the data based on columns 'type' and 'region' passed into the list and compute the average price

by_type_year = data.groupby(['type','year'])['AveragePrice'].mean()
print(by_type_year)

# COMMAND ----------

# TASK 4 >>>> Group the data based on columns 'type', 'year' and 'region' passed into the list
# and compute how many kg of Large Hass Avocados have been sold in total. 
# Assign the result to the variable by_year.

# COMMAND ----------

# MAGIC %md
# MAGIC When we are using the `.groupby()`, the resulting object will be slightly different from a standard Pandas dataframe. You can see it in the print statement and how "type" and "year" are nicely printed. 
# MAGIC 
# MAGIC If we would like to work with the resulting object further, we should reset its row index by using `reset_index()` and convert it into a regular dataframe.

# COMMAND ----------

# Reset the index using .reset_index() method and create a DataFrame
our_df = pd.DataFrame(by_year).reset_index()
print(our_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Aggregate function (ADVANCED)
# MAGIC 
# MAGIC ![](https://keytodatascience.com/wp-content/uploads/2020/04/image-1.png)
# MAGIC 
# MAGIC [image source](https://keytodatascience.com/groupby-pandas-python/)

# COMMAND ----------

# MAGIC %md
# MAGIC The aggregation method `agg()`.
# MAGIC 
# MAGIC - it is an alias for aggregate
# MAGIC - it is used to pass a function or list of functions to be applied on a series or even each element of a series separately
# MAGIC 
# MAGIC This can be done by passing the columns and functions within a dictionary like this:
# MAGIC 
# MAGIC `our_dataset.agg({'First_column' : ['max', 'min'], 'Second_column' : ['mean', 'median']})`

# COMMAND ----------

# Compute maximum and minimum values for column 'Total Volume' and minimum and mean values for column 'Small Bags' using .agg()
data.agg({'Total Volume' : ['max', 'min'], 'Small Bags' : ['min', 'mean']})

# COMMAND ----------

# MAGIC %md
# MAGIC We can pass `.agg()` also to our grouped object and compute statistics for selected column.

# COMMAND ----------

# Group the data based on the two columns 'region' and 'type'
# Compute aggregations 'min','max' and 'mean' for 'AveragePrice'
grouped = data.groupby(['region','type']).agg({'AveragePrice':['min','max','mean']})

# COMMAND ----------

grouped

# COMMAND ----------

# MAGIC %md
# MAGIC - within `agg()` we can have our custom function along with computing aggregation

# COMMAND ----------

# Write a function to compute 95th percentile on desired column using .quantile(0.95)
def percentile_95(column):
    return column.quantile(0.95)

# COMMAND ----------

# TASK 5 - HARD >>>> Get 95th percentile and mean values for columns: 'Small Bags','Large Bags','XLarge Bags' from DataFrame data, using .agg()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Bonus Task (HARD)
# MAGIC 
# MAGIC `groupby()` can be useful when we want to look at the proportion of avocado's type. We would like to see what percentages of conventional and organic avocados have been sold. For example: 97 % and 3%.
# MAGIC 
# MAGIC To reach this result:
# MAGIC - Group the data by 'type' and obtain sums on the 'Total Volume' column, assign result to `volume_by_type`
# MAGIC - Divide `volume_by_type` by the sum of all avocados. Assign the result to the variable `proportion`.
# MAGIC - Print the proportion and optionally multiply it by 100 to obtain a figure in percentage

# COMMAND ----------

# TASK 6.1 >>>> Group data based on their types and compute count of the Total Volume 

# COMMAND ----------

# TASK 6.2 >>>> Compute the proportion of the avocado's type

# COMMAND ----------

# TASK 6.3 >>>> Print the output multiply by 100

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Appendix
# MAGIC 
# MAGIC Data Source: https://www.kaggle.com/neuromusic/avocado-prices
# MAGIC 
# MAGIC License: Database: Open Database, Contents: © Original Authors
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. Source: https://github.com/zatkopatrik/authentic-data-science

# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Database-style DataFrame Merges
# MAGIC 
# MAGIC Merge operations combine DataFrames on common columns or indices.

# COMMAND ----------

import pandas as pd
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# Run this code
data_1 = pd.DataFrame({'key':['A','B','C','B','E','F','A','H','A','J'],
                      'values_1': range(10)})
print(data_1)

# COMMAND ----------

# Run this code
data_2 = pd.DataFrame({'key':['A','B','C'],
                       'values_2':range(3)})
print(data_2)

# COMMAND ----------

# MAGIC %md
# MAGIC Our first DataFrame **data_1** has multiple rows with keys 'A' and 'B', whereas DataFrame **data_2** has only 1 row for each value in the `key` column. This is an example of `many-to-one` $^{1}$ merge situation.
# MAGIC 
# MAGIC By merging these 2 dataframes we obtain following result:

# COMMAND ----------

# Merge data_1 and data_2
pd.merge(data_1, data_2)

# COMMAND ----------

# MAGIC %md
# MAGIC Our DataFrames have the same column `key ` and in this case `.merge()` uses the overlapping column named as `keys` to join on. However it is a good practice to specify explicitly the `key` column like this:

# COMMAND ----------

# Merge data_1 and data_2, specify key column
pd.merge(data_1, data_2, on = 'key')

# COMMAND ----------

# MAGIC %md
# MAGIC As you can notice 'E', 'F', 'H', 'J'  and associated data are missing from the result. It is because `merge()` acts with 'inner' merge (join) by default. However, we can explicitly specify it using `how = 'inner'`
# MAGIC 
# MAGIC **inner join** (or inner merge) keeps only those values that have common key in both DataFrames, in our case 'A', 'B' and 'C'. 
# MAGIC 
# MAGIC 
# MAGIC Other possible options are:
# MAGIC  
# MAGIC - **left join** (left outer join)
# MAGIC 
# MAGIC We specify `how = 'left'`: keeps each row from the left DataFrame and only those from the right DataFrame that match. Non-matching values are replaced with NaNs.
# MAGIC 
# MAGIC 
# MAGIC - **right join** (right outer join)
# MAGIC 
# MAGIC We specify `how = 'right'`: it is the opposite of left join. Non-matching values are filled with NaNs as well.
# MAGIC 
# MAGIC - **outer** (full outer join)
# MAGIC 
# MAGIC We specify `how = 'outer'`: it takes the union of the keys and applies both left and right join
# MAGIC 
# MAGIC Run the following code to see these merging strategies $^{2}$.

# COMMAND ----------

# Run this to print merging strategies
Image(filename='Images/merging.png')

# COMMAND ----------

# Merge the DataFrames data_1 and data_2 with left join
pd.merge(data_1, data_2, on = 'key', how = 'left')

# COMMAND ----------

# TASK 1 >>>> Merge the dataframes data_1 and data_2 on 'key', specify right join
pd.merge(data_1, data_2, on = 'key', how = 'right')

# COMMAND ----------

# TASK 2 >>>> Merge the dataframes data_1 and data_2 on 'key', specify full outer join
pd.merge(data_1, data_2, on = 'key', how = 'outer')

# COMMAND ----------

# MAGIC %md
# MAGIC If the key column names are different in each DataFrame object, we can specify them separately.
# MAGIC 
# MAGIC - for the left DataFrame: `left_on`
# MAGIC - for the right DataFrame: `right_on`

# COMMAND ----------

# Run this code
data_3 = pd.DataFrame({'key_left': ['E','F','G','H','I','J'],
                       'values': range(6)})
print(data_3)

# COMMAND ----------

# Run this code
data_4 = pd.DataFrame({'key_right': ['D','E','F','G'],
                       'values_2': range(4)})
print(data_4)

# COMMAND ----------

# Merge the DataFrames data_3 and data_4, specify left and right keys to join on
# Specify inner join 
pd.merge(data_3, data_4, left_on= 'key_left', right_on= 'key_right', how = 'inner')

# COMMAND ----------

# Run this code
df_1 = pd.DataFrame({'key': ['red','black','yellow','green','black','pink','white','black'],
                     'values': range(8)})
print(df_1)

# COMMAND ----------

# Run this code
df_2 = pd.DataFrame({'key': ['white','pink','gray','yellow','black','black','black'],
                     'values': range(7)})
print(df_2)

# COMMAND ----------

# Merge df_1 and df_2 on 'key', specify left join
pd.merge(df_1, df_2, on = 'key', how = 'left')

# COMMAND ----------

# MAGIC %md
# MAGIC This is `many-to-many` $^{1}$ join situation which creates **Cartesian product** of the rows. In the result we can see we have 9 'black' rows. It is because there is 3 'black' rows in the left DataFrame df_1 and 3 'black' rows in the right DataFrame df_2, so in the result we have every combination of rows where the key is equal to 'black'
# MAGIC 
# MAGIC As you can see `merge()` automatically rename column as 'values_x' and 'values_y' to distinguish where the values belong to. We can explicitly specify these column's names with suffixes option. We only need to pass desired names into the list like this: `suffixes=['_from_df1', '_from_df2']`.

# COMMAND ----------

# Merge df_1 and df_2 on 'key', specify left join
# Set parameter suffixes=[]
pd.merge(df_1, df_2, on = 'key', how = 'left', suffixes=['_from_df1', '_from_df2'])

# COMMAND ----------

# MAGIC %md
# MAGIC - if we want to merge with multiple keys we have to pass a list of columns names:

# COMMAND ----------

# Run this code
df_3 = pd.DataFrame({'key_1':['apple','banana','coconut','pineapple','strawberry'],
                     'key_2':['yes','maybe','maybe','yes','no'],
                     'values_1': range(5)})
print(df_3)

# COMMAND ----------

# Run this code
df_4 = pd.DataFrame({'key_1':['apple','banana','coconut','strawberry','strawberry'],
                     'key_2':['no','maybe','yes','no','no'],
                     'values_1': range(5)})
print(df_4)

# COMMAND ----------

# Merge DataFrames df_3 and df_4 on column keys 'key_1' and 'key_2' passed within the list and specify inner join
pd.merge(df_3, df_4, on = ['key_1', 'key_2'], how = 'inner')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Advanced and Alternative Methods (READ-AND-PLAY)
# MAGIC If you are familiar and fine with using the merge method, you should be good to go. You might however stumble also upon some alternative, sometimes more complex methods, for doing similar things. Let's read through those.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Merging DataFrames on the Index
# MAGIC 
# MAGIC Our key(s) columns for merging can be found in a DataFrame as an index. In this case we can use the parameters `left_index = True` or `right_index = True` (or both) to indicate that the index should be used as the merge key.
# MAGIC 
# MAGIC - left_index : bool (default False)
# MAGIC    - if True will choose index from left DataFrame as join key
# MAGIC - right_index : bool (default False)
# MAGIC    - if True will choose index from right DataFrame as join key

# COMMAND ----------

# Run this code
students = [(1, 'Robert', 30, 'Slovakia', 26),
           (2, 'Jana', 29, 'Sweden' , 27),
           (3, 'Martin', 31, 'Sweden', 26),
           (4, 'Kristina', 26,'Germany' , 30),
           (5, 'Peter', 33, 'Austria' , 22),
           (6, 'Nikola', 25, 'USA', 23),
           (7, 'Renato', 35, 'Brazil', 26)]

students_1 = pd.DataFrame(students, columns= ['student_id', 'first_name', 'age', 'city', 'score'])
students_1.set_index('student_id', inplace = True)
print(students_1)

# COMMAND ----------

# Run this code
programs = [(1, 'Data Science', 3),
            (2, 'Data Analyst', 1),
            (3, 'Microbiology', 4),
            (4, 'Art History', 2),
            (5, 'Chemistry', 5),
            (6, 'Economics', 4),
            (7, 'Digital Humanities', 2)]

programs_1 = pd.DataFrame(programs, columns= ['student_id', 'study_program', 'grade'])
programs_1.set_index('student_id', inplace = True)
print(programs_1)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, DataFrames `students_1` and `programs` share the same column 'student_id' that is set as an index.

# COMMAND ----------

# Merge students_1 and programs on 'student_id' by passing `left_index = True` and `right_index = True`

merged_df = pd.merge(students_1, programs_1, how = 'inner', left_index = True, right_index = True)
print(merged_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Pandas `.join()`

# COMMAND ----------

# MAGIC %md
# MAGIC - it is an object method function - it means that it enables us to specify only 1 DataFrame to be joined to the DataFrame from which you call `.join()` on
# MAGIC - by default it performs left join
# MAGIC - by default it **joins on indices**

# COMMAND ----------

# Join students_1 and programs_1 
joined_df = students_1.join(programs_1)
print(joined_df)

# COMMAND ----------

# Run this code, please
programs_1.reset_index(inplace = True)
students_1.reset_index(inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC If we want to join DataFrames that have overlapping column keys, we need to specify parameter `lsuffix` and `rsuffix`.

# COMMAND ----------

# Join students_1 and programs_1
# Specify suffixes for both DataFrames
joined_df = students_1.join(programs_1, lsuffix = '_left', rsuffix = '_right')
print(joined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.3 Pandas `.concat()`
# MAGIC 
# MAGIC -concatenate function combines DataFrames across rows or columns 
# MAGIC - by default performs outer join, but we can specify inner join by setting `join = 'inner'`
# MAGIC 
# MAGIC - by default works along axis = 0 (rows)
# MAGIC 
# MAGIC `pd.concat([df1, df2])`
# MAGIC 
# MAGIC - we can pass axis = 1 to concatenate along columns 
# MAGIC 
# MAGIC `pd.concat([df1, df2], axis = 1)`

# COMMAND ----------

# Concatenate students_1 and programs_11 along the rows
concat_by_rows = pd.concat([students_1, programs_1])
print(concat_by_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC Column names in DataFrames students_1 and programs_1 are not the same. As we can see in the exapmle above, by default, those columns have been also added on the result and NaN values have been filled in.
# MAGIC 
# MAGIC We can also create a hierarchical index on the concatenation axis, when we use argument `keys = ['key1','key2','key3','key_n'...]`.

# COMMAND ----------

# Concatenate the DataFrames programs_1 and students_1 along the rows
# Set keys argument on columns 'student_id' and 'study_program'
conc = pd.concat([programs_1, students_1], keys = ['student_id','study_program'] )
print(conc)

# COMMAND ----------

# Concatenate df_3 and df_4 along the rows
concat_df = pd.concat([df_3, df_4])
print(concat_df)

# COMMAND ----------

# MAGIC %md
# MAGIC DataFrames df_3 and df_4 have the same column names 'key_1' and 'key_2'. Therefore the indices are repeating when tha DataFrames are stacked. If you want to have 0-based index, you'll need to set parameter `ignore_index = True` within `.concat()` function

# COMMAND ----------

# Concatenate df_3 and df_4 along the rows
# Set the parameter `ignore_index = True`
concat_df_2 = pd.concat([df_3, df_4], ignore_index = True)
print(concat_df_2)

# COMMAND ----------

# Concatenate students_1 and programs_11 along the columns
concat_by_columns = pd.concat([students_1, programs_1], axis = 1)
print(concat_by_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Pandas `.append()`
# MAGIC 
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC - this method is a shortcut to `.concat()`
# MAGIC - it is used to add elements to the existing objects

# COMMAND ----------

# Call .append() method on students_1 and pass programs_1 as an argument within the method
appended_df = students_1.append(programs_1)
print(appended_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We can append new rows to our existing DataFrames as well. We simply create a new Series (or a dictionary) with items and associated  indeces (column name) and append it to existing DataFrame using `.append()` method. New row can be appended only if the `ignore_index = True` argument is set within the method.

# COMMAND ----------

# Run this code 
# It will create new_row which we'll append to the students_1 
new_row = pd.Series([8, 'Renata', 37, 'Czech Republic', 22],
                    index = ['student_id', 'first_name', 'age', 'city', 'score'])

# COMMAND ----------

# Append new_row Series to the students_1 DataFrame 
# Set argument 'ignore_index = True'
students_1 = students_1.append(new_row, ignore_index = True)
print(students_1)

# COMMAND ----------

# MAGIC %md
# MAGIC We can use `.append()` method to add both single element and list to existing list.

# COMMAND ----------

# Run this code
list_1 = ['Today', 'is', 'a', 'good']

# COMMAND ----------

# Append word 'day' to list_1
list_1.append('day')
print(list_1)

# COMMAND ----------

# Run this code
new_list = [10,20,30,40,50]

# COMMAND ----------

# Append new_list to list_1 
list_1.append(new_list)
print(list_1)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. References
# MAGIC 
# MAGIC 1. Wes Mckinney (2013). Python for Data Analysis. (First ed.). California: O'Reilly Media, Inc.
# MAGIC 
# MAGIC 2. Medium. Merging DataFrames with pandas. [ONLINE] Available at: https://medium.com/swlh/merging-dataframes-with-pandas-pd-merge-7764c7e2d46d. [Accessed 14 September 2020].
# MAGIC 
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. Source: https://github.com/zatkopatrik/authentic-data-science

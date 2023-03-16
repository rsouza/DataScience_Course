# Databricks notebook source
# MAGIC %md
# MAGIC ### Simple examples of data imputation with scikit-learn
# MAGIC #### (read and play)

# COMMAND ----------

import numpy as np
import pandas as pd
from io import StringIO

# COMMAND ----------

# MAGIC %md
# MAGIC Creating some data with missing values

# COMMAND ----------

csvdata = '''
A,B,C,D,E
1,2,3,4,
5,6,,8,
0,,11,12,13
'''

df = pd.read_csv(StringIO(csvdata))
df

# COMMAND ----------

# MAGIC %md
# MAGIC Radical choice: delete whole column

# COMMAND ----------

df.drop(["E"], axis=1, inplace=True)
df

# COMMAND ----------

# MAGIC %md Recreating

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))
df

# COMMAND ----------

# MAGIC %md
# MAGIC Less Radical: delete rows with missing values on "C" column

# COMMAND ----------

df.dropna(axis=0, how='any', thresh=None, subset=["C"], inplace=True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC If you do not specify the columns, it will delete every row with any missing value

# COMMAND ----------

df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df

# COMMAND ----------

# MAGIC %md
# MAGIC Imputing with scikit-learn

# COMMAND ----------

from sklearn.impute import SimpleImputer

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))
df

# COMMAND ----------

# MAGIC %md
# MAGIC Imputing mean values

# COMMAND ----------

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(df["C"].values.reshape(-1,1))
df["C"] = imp.transform(df["C"].values.reshape(-1,1))
df

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))
df

# COMMAND ----------

# MAGIC %md
# MAGIC Imputing a constant value

# COMMAND ----------

imp = SimpleImputer(missing_values=np.nan, fill_value=200, strategy='constant')
imp.fit(df["C"].values.reshape(-1,1))
df["C"] = imp.transform(df["C"].values.reshape(-1,1))
df

# COMMAND ----------

df = pd.read_csv(StringIO(csvdata))
df

# COMMAND ----------

# MAGIC %md
# MAGIC Interactive imputing (experimental)

# COMMAND ----------

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# COMMAND ----------

imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(df)
columns = df.columns
df = pd.DataFrame(imp_mean.transform(df), columns=columns)
df

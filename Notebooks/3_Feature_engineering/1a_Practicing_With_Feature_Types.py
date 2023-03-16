# Databricks notebook source
# MAGIC %md
# MAGIC # Exploring Feature Types
# MAGIC 
# MAGIC In this notebook we will be using Titanic dataset containing 891 records of passengers on board and 12 features such as their age, travel class and the ticket's fare. One step at a time we'll inspect continuous and categorical feature types with the help of visualizations. 
# MAGIC 
# MAGIC Firstly, we import necessary libraries and load the data using Pandas.

# COMMAND ----------

# Importing Pandas, Matplotlib and Seaborn libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")

# COMMAND ----------

# Load the dataset 'Data/titanic_data.csv' and store it to variable data
data = pd.read_csv('Data/titanic_data.csv')

# Get first 10 rows of the data
data.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the first ten rows and read through the explanation below to understand what each feature represents.
# MAGIC 
# MAGIC - passengerID
# MAGIC - Name
# MAGIC - Sex
# MAGIC - Age
# MAGIC - Survived - not survived = 0, survived = 1, (target feature)
# MAGIC - Pclass - ticket class = 1st, 2nd, 3rd
# MAGIC - SibSp - number of siblings or spouses aboard the Titanic
# MAGIC - Parch - number of parents or children aboard the Titanic
# MAGIC - Ticket - ticket number
# MAGIC - Fare - ticket fare
# MAGIC - Cabin - cabin number
# MAGIC - Embarked - port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Numerical Features
# MAGIC 
# MAGIC ## 1.1 Unbounded continuous type
# MAGIC One of the numerical feature is 'Fare' representing the price that a passenger paid for the ticket. The price starts at zero and continuously increases within a specific range. This is the example of unbounded continuous feature type because the number can takes any value including decimal numbers with a fractional part. 

# COMMAND ----------

# Creating a histogram of 'Fare' 
sns.histplot(data = data, x = 'Fare', binwidth = 15)
plt.title("Distribution of passenger's fare");

# COMMAND ----------

# MAGIC %md
# MAGIC The next numerical continuous feature is 'Age'. The other characteristic of a continuous feature is that it can be measured, as we can measure the age in years, for example. Let's create a boxplot to see its distribution. 

# COMMAND ----------

# Creating a boxplot of 'Age'
sns.boxplot(data=data, x = 'Age',color ='r');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Unbounded discrete type
# MAGIC 
# MAGIC The feature 'SibSp' represents family relations as siblings or spouses. We can treat this feature as _discrete_ since the number is always some "isolated" value - logically, you can't report that you have 2.5 sisters.
# MAGIC 
# MAGIC When we call `value_counts()` function on the 'SibSp' column, we get all of the unique values along with corresponding counts. 

# COMMAND ----------

# Get the counts of unique values of 'SibSp'
data['SibSp'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC When we visualize such a discrete feature, let's say with a countplot, each bar represents a unique integer value with respective count of records. There is no overlapping between these fixed values because they are counted, not measured.

# COMMAND ----------

# Creating a countplot of 'SibSp'
sns.countplot(data = data, x = 'SibSp', color = 'darkgreen')
plt.title('Number of parents or children');

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Categorical Features
# MAGIC 
# MAGIC Categorical features contain a set of distinct categories (also called _labels_), while each category can takes only one limited and fixed value. As you saw in the preceding lesson, categorical features are divided into two 'subtypes' based on whether their values have order or not. 
# MAGIC 
# MAGIC ## 2.1 Ordinal categorical features
# MAGIC 
# MAGIC Let's take as an example 'Pclass' feature that holds information on the ticket class each passenger paid for: 1st, 2nd or 3rd class. These travel classes reflect socioeconomic status of the passengers aboard. They have the specific order and can be related to the target feature. Were the wealthiest passengers travelling in the first class more likely to survive? Or is class of no importance in terms of survival? 
# MAGIC 
# MAGIC Ordinal categorical features can be either numeric values or labels. Still, it would be nice to keep the information about order of values and present it to the predictive model. This would be the additional information that can make the model's predictions better.

# COMMAND ----------

# Get the counts of unique values of 'Pclass'
data.Pclass.value_counts()

# COMMAND ----------

# Creating a countplot of 'Pclass'
sns.countplot(data = data, x = 'Pclass', color = 'violet');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Non-ordinal categorical features
# MAGIC The second 'subtype' of categorical features are non-ordinal features. Let's take as an example the feature 'Embarked'. It contains information about port of embarkation, namely Cherbourg, Queenstown and Southampton. So these are specific categories without any order or relationship among them.

# COMMAND ----------

# Get the categories of 'Embarked' 
data['Embarked'].value_counts()

# COMMAND ----------

# Creating a countplot of 'Embarked'
sns.countplot(data = data, x = 'Embarked', color = 'lightblue');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Binary categorical features
# MAGIC 
# MAGIC Binary categorical features are a special type of categorical features which take only 2 values. For example, the 'Survived' feature in our dataset contains just 2 values: 0 and 1.

# COMMAND ----------

# Get the unique values of 'Survived' column
data['Survived'].unique()

# COMMAND ----------

# Creating a countplot of 'Survived' 
sns.countplot(data = data, x = 'Survived', color = 'violet');

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix
# MAGIC 
# MAGIC Data source: https://www.kaggle.com/hesh97/titanicdataset-traincsv
# MAGIC 
# MAGIC Data license: CC0: Public Domain
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science) 

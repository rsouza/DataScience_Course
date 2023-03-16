# Databricks notebook source
# MAGIC %md
# MAGIC # Data Science workflow  
# MAGIC 
# MAGIC In this sequence of notebooks we will exemplify the inner steps in the Data Science workflow.  
# MAGIC We are not going to discuss the business requirements and deployment strategies, but just the phases below:
# MAGIC 
# MAGIC ### I - Exploratory Data Analysis (this notebook)  
# MAGIC ##### II - Feature Engineering and Selection 
# MAGIC ##### III - Modeling  
# MAGIC ##### IV - Evaluation  
# MAGIC 
# MAGIC This notebook will cover the Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## I - Exploratory Data Analysis  
# MAGIC 
# MAGIC Exploratory Data Analysis is a set of techniques developed by John Wilder Tukey in 1970. The philosophy behind this approach was to examine the data before building a model.  
# MAGIC John Tukey encouraged statisticians to explore the data, and possibly formulate hypotheses that could lead to new data collection and experiments.  
# MAGIC 
# MAGIC Today data scientists and analysts spend most of their time in Data Wrangling and Exploratory Data Analysis also known as EDA. But what is this EDA and why is it so important? 
# MAGIC Exploratory Data Analysis (EDA) is a step in the data science workflow, where a number of techniques are used to better understand the dataset being used.
# MAGIC 
# MAGIC ‘Understanding the dataset’ can refer to a number of things including but not limited to…
# MAGIC 
# MAGIC + Get maximum insights from a data set
# MAGIC + Uncover underlying structure
# MAGIC + Extracting important variables and leaving behind useless variables
# MAGIC + Identifying outliers, anomalies, missing values, or human error
# MAGIC + Understanding the relationship(s), or lack thereof, between variables
# MAGIC + Testing underlying assumptions
# MAGIC + Ultimately, maximizing your insights in a dataset and minimizing potential error that may occur later in the process

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Let's see how exploratory data analysis is regarded in CRISP-DM and CRISP-ML:

# COMMAND ----------

# MAGIC %md
# MAGIC ## CRISP-DM
# MAGIC 
# MAGIC The CRoss Industry Standard Process for Data Mining ([CRISP-DM](https://www.datascience-pm.com/crisp-dm-2/)) is a process model that serves as the base for a data science process.  
# MAGIC It has six sequential phases:
# MAGIC 
# MAGIC + Business understanding – What does the business need?
# MAGIC + Data understanding – What data do we have / need? Is it clean?
# MAGIC + Data preparation – How do we organize the data for modeling?
# MAGIC + Modeling – What modeling techniques should we apply?
# MAGIC + Evaluation – Which model best meets the business objectives?
# MAGIC + Deployment – How do stakeholders access the results?
# MAGIC 
# MAGIC 
# MAGIC ![CRISP-DM Process](https://miro.medium.com/max/736/1*0-mnwXXLlMB_bEQwp-706Q.png)

# COMMAND ----------

# MAGIC %md
# MAGIC The machine learning community is still trying to establish a standard process model for machine learning development. As a result, many machine learning and data science projects are still not well organized. Results are not reproducible.  
# MAGIC In general, such projects are conducted in an ad-hoc manner. To guide ML practitioners through the development life cycle, the Cross-Industry Standard Process for the development of Machine Learning applications with Quality assurance methodology ([CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml)) was recently proposed.  
# MAGIC 
# MAGIC There is a particular order of the individual stages. Still, machine learning workflows are fundamentally iterative and exploratory so that depending on the results from the later phases we might re-examine earlier steps.

# COMMAND ----------

# MAGIC %md
# MAGIC ## CRISP-ML
# MAGIC 
# MAGIC ![CRISP-ML Process](https://ml-ops.org/img/crisp-ml-process.jpg)  
# MAGIC [Source](https://ml-ops.org/content/crisp-ml)

# COMMAND ----------

# MAGIC %md
# MAGIC If we explode the EDA phase in each of the previous frameworks, we would have something like this:
# MAGIC 
# MAGIC ![EDA](https://www.researchgate.net/publication/329930775/figure/fig3/AS:873046667710469@1585161954284/The-fundamental-steps-of-the-exploratory-data-analysis-process_W640.jpg)  
# MAGIC [Source](https://www.researchgate.net/publication/329930775_A_comprehensive_review_of_tools_for_exploratory_analysis_of_tabular_industrial_datasets)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Starting the EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Import libraries

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #https://towardsdatascience.com/a-major-seaborn-plotting-tip-i-wish-i-had-learned-earlier-d8209ad0a20e
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Load Dataset and distinguish attributes

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.1 - Visually inspecting the dataset

# COMMAND ----------

df = pd.read_csv('Data/Automobile_data.csv')
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.2 - Checking columns and data types

# COMMAND ----------

# df.columns
df.info(verbose=True, show_counts=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2.3 - Checking basic statistics - first insight on distributions

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### At this moment, you look for columns that shall be transformed/converted later in the workflow.

# COMMAND ----------

print(df.select_dtypes(include='number').columns)
print(df.select_dtypes(include='object').columns)
print(df.select_dtypes(include='category').columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Check for missing values

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC It seems there are no missing values, but that may be misleading. Let's explore a bit more:

# COMMAND ----------

#Checking for wrong entries like symbols -,?,#,*,etc.
for col in df.columns:
    print('{} : {}'.format(col, df[col].unique()))

# COMMAND ----------

# MAGIC %md
# MAGIC There are null values in our dataset in form of ‘?’. Pandas is not recognizing them so we will replace them with `np.nan`.

# COMMAND ----------

for col in df.columns:
    df[col].replace({'?': np.nan},inplace=True)
    
df.info()

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Visualizing the missing values  
# MAGIC Now the missing values are identified in the dataframe. With the help of `heatmap`, we can see the amount of data that is missing from the attribute. With this we can make decisions whether to drop these missing values or to replace them. Usually dropping the missing values is not advisable but sometimes it may be helpful.

# COMMAND ----------

plt.figure(figsize=(12,10))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')

# COMMAND ----------

# MAGIC %md
# MAGIC Now observe that there are many missing values in 'normalized_losses' while other columns have fewer missing values. We can’t drop the 'normalized_losses' column as it may be important for our prediction.  
# MAGIC We can also use the **missingno** libray for a better evaluation of the missing values. First we can check the quantity and how they distribute among the rows:

# COMMAND ----------

#!pip install missingno

# COMMAND ----------

import missingno as msno

# COMMAND ----------

msno.bar(df)

# COMMAND ----------

msno.matrix(df)

# COMMAND ----------

# MAGIC %md
# MAGIC The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another

# COMMAND ----------

msno.heatmap(df)

# COMMAND ----------

# MAGIC %md
# MAGIC The dendrogram allows you to more fully correlate variable completion, revealing trends deeper than the pairwise ones visible in the correlation heatmap.

# COMMAND ----------

msno.dendrogram(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2. Replacing the missing values
# MAGIC We will be replacing these missing values with mean because the number of missing values is not great (we could have used the median too).  
# MAGIC Later, in the data preparation phase, we will learn other imputation techniques.

# COMMAND ----------

df.select_dtypes(include='number').head()

# COMMAND ----------

df.select_dtypes(include='object').head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's transform the mistaken datatypes for numeric values and fill with the mean, using the strategy we have chosen.

# COMMAND ----------

num_col = ['normalized-losses', 'bore',  'stroke', 'horsepower', 'peak-rpm','price']
for col in num_col:
    df[col] = pd.to_numeric(df[col])
    df[col].fillna(df[col].mean(), inplace=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Checking Data Distributions
# MAGIC 
# MAGIC This is the most important step in EDA. 
# MAGIC - This step will decide how much insight you can get.
# MAGIC - Checking the distributions is fundamental for feature selection and the modeling phase
# MAGIC - This step varies from person to person in terms of their questioning ability. 
# MAGIC 
# MAGIC Let's check the univariate and bivariate distributions and correlation between different variables, this will give us a roadmap on how to proceed further.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Univariate Analysis  
# MAGIC 
# MAGIC The goal here is to check the distribution of numeric and categorical variables (more about this later in the course)  
# MAGIC We can quickly check the distributions of every numeric column:

# COMMAND ----------

numeric_cols = df.select_dtypes(include='number').columns
numeric_cols

# COMMAND ----------

for col in numeric_cols:
    plt.figure(figsize=(18,5))
    plt.subplot(1,2,1)
    #sns.distplot(df[col])
    sns.histplot(df[col], kde=True)
    plt.subplot(1,2,2)
    sns.boxplot(x=col, data=df)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.1.1 - Analizing distributions on numerical variables - Spotting outliers
# MAGIC 
# MAGIC ![Outliers](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/PH717-QuantCore/PH717-Module6-RandomError/Normal%20Distribution%20deviations.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Assuming the data would follow a normal distribution, we can choose some of the graphs to examine the data in more detail:

# COMMAND ----------

#set the style we wish to use for our plots
sns.set_style("darkgrid")

#plot the distribution of the DataFrame "Price" column
plt.figure(figsize=(8,12))
#sns.histplot(df['price'])
sns.displot(df['peak-rpm'], kde=True, bins=50, height=8, aspect=2)  

# COMMAND ----------

fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x="peak-rpm", data=df, ax=ax)

# COMMAND ----------

# MAGIC %md
# MAGIC We will not treat outliers during Exploratory Data Analysis, but we will get back to them in the Data Preparation phase.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.1.2 - Analizing distributions on categorical variables

# COMMAND ----------

# MAGIC %md
# MAGIC Although it is not one of the recommended plots, we can always use the pie plots in special situations:

# COMMAND ----------

fig, ax = plt.subplots(figsize=(8,8))
plt.pie(df["body-style"].value_counts(sort=False), labels=df["body-style"].unique())
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Barplots with frequencies can be created in Matplotlib.

# COMMAND ----------

df["body-style"].value_counts().plot(kind="bar", figsize=(10,6))

# COMMAND ----------

# MAGIC %md
# MAGIC There is no need to separately calculate the count when using the `sns.countplot()` function

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(df["body-style"], ax=ax) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Bivariate Analysis  
# MAGIC 
# MAGIC Now we want to check the relationships between pais of variables. We can start by drawing a pairplot and a correlation plot.

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.pairplot(df.select_dtypes(include='number'))

# COMMAND ----------

# MAGIC %md
# MAGIC The pairplot can help us gaining quick insights on the correlations of variables, but can get cluttered if we have many features.  
# MAGIC We can also try the heatmap of correlations:

# COMMAND ----------

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), cbar=True, annot=True, cmap='Blues')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Positive Correlation  
# MAGIC + 'Price' – 'wheel-base', 'length', 'width', 'curb_weight', 'engine-size', 'bore', 'horsepower'  
# MAGIC + 'wheel-base' – 'length', 'width', 'height', 'curb_weight', 'engine-size', 'price'  
# MAGIC + 'horsepower' – 'length', 'width', 'curb-weight', 'engine-size', 'bore', 'price'  
# MAGIC + 'Highway mpg' – 'city-mpg'  
# MAGIC 
# MAGIC ##### Negative Correlation  
# MAGIC + 'Price' – 'highway-mpg', 'city-mpg'  
# MAGIC + 'highway-mpg' – 'wheel base', 'length', 'width', 'curb-weight', 'engine-size', 'bore', 'horsepower', 'price'  
# MAGIC + 'city' – 'wheel base', 'length', 'width', 'curb-weight', 'engine-size', 'bore', 'horsepower', 'price'  
# MAGIC 
# MAGIC This heatmap has given us great insights into the data.  
# MAGIC Now let us apply domain knowledge and ask the questions which will affect the price of the automobile.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.2.1 - Checking some columns in more detail  
# MAGIC We can draw a vertical boxplot grouped by a categorical variable:

# COMMAND ----------

fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x="fuel-type", y="horsepower", data=df, ax=ax)

# COMMAND ----------

# MAGIC %md
# MAGIC And even add a third component:  
# MAGIC https://seaborn.pydata.org/tutorial/categorical.html

# COMMAND ----------

#sns.catplot(x="fuel-type", y="horsepower", hue="num-of-doors", kind="box", data=df, height=8, aspect=2)
sns.catplot(x="fuel-type", y="horsepower", hue="num-of-doors", kind="violin", inner="stick", split=True, palette="pastel", data=df, height=8, aspect=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Asking questions based on the analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Try to ask questions related to independent variables and the target variable.  
# MAGIC Example questions related to this dataset could be:  
# MAGIC 
# MAGIC + How does 'fuel-type' affect the price of the car?   
# MAGIC + How does the 'horsepower' affect the price?  
# MAGIC + What is the relationship between 'engine-size' and 'price'?  
# MAGIC + How does 'highway-mpg' affects 'price'?  
# MAGIC + What is the relation between no. of doors and 'price'?

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.1 How does 'fuel_type' will affect the price?  
# MAGIC 
# MAGIC Let's compare categorical data with numerical data. We are going to use a catplot from Seaborn, but there are other options for categorical variables:  
# MAGIC https://seaborn.pydata.org/tutorial/categorical.html

# COMMAND ----------

plt.figure(figsize=(12,10))
#https://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot
sns.catplot(x='fuel-type',y='price', data=df, height=8)
plt.xlabel('Fuel Type')
plt.ylabel('Price')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 How does the horsepower affect the price?  
# MAGIC 
# MAGIC Matplotlib and Seaborn have very nice graphs to visualize numerical relationships:  
# MAGIC https://seaborn.pydata.org/tutorial/relational.html  
# MAGIC https://matplotlib.org/stable/gallery/index.html

# COMMAND ----------

plt.figure(figsize=(12,10))
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html  
plt.scatter(x='horsepower',y='price', data=df)
plt.xlabel('Horsepower')
plt.ylabel('Price')

# COMMAND ----------

#https://seaborn.pydata.org/generated/seaborn.jointplot.html

sns.jointplot(x='horsepower',y='price', data=df)
sns.jointplot(x='horsepower',y='price', data=df, kind='hex')

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that most of the horsepower values lie between 50-150 with a price mostly between 5000-25000. There are outliers as well (between 200-300).  
# MAGIC Let’s see a count between 50-100 i.e univariate analysis of horsepower.

# COMMAND ----------

plt.figure(figsize=(12,10))
#https://seaborn.pydata.org/generated/seaborn.histplot.html
sns.histplot(df.horsepower,bins=10)

# COMMAND ----------

# MAGIC %md
# MAGIC The average count between 50-100 is 50 and it is positively skewed.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.3 What is the relation between engine-size and price?

# COMMAND ----------

plt.figure(figsize=(12,10))
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html  
plt.scatter(x='engine-size',y='price',data=df)
plt.xlabel('Engine size')
plt.ylabel('Price')

# COMMAND ----------

sns.jointplot(x='engine-size',y='price', data=df, kind='reg')
sns.jointplot(x='engine-size',y='price', data=df, kind='kde')

# COMMAND ----------

# MAGIC %md
# MAGIC We can observe that the pattern is similar to horsepower vs price.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4 How does highway_mpg affects price?

# COMMAND ----------

plt.figure(figsize=(12,10))
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html  
plt.scatter(x='highway-mpg',y='price',data=df)
plt.xlabel('Higway mpg')
plt.ylabel('Price')

# COMMAND ----------

# MAGIC %md
# MAGIC We can see price decreases with an increase in 'higway-mpg'.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.5 What is the relation between no. of doors and price?  
# MAGIC 
# MAGIC Let us first check the number of doors.

# COMMAND ----------

# Unique values in num_of_doors
df["num-of-doors"].value_counts().plot(kind="bar", figsize=(10,6))

# COMMAND ----------

plt.figure(figsize=(12,8))
#https://seaborn.pydata.org/generated/seaborn.boxplot.html
sns.boxplot(x='price', y='num-of-doors',data=df)

# COMMAND ----------

# MAGIC %md
# MAGIC With this boxplot we can conclude that the average price of a vehicle with two doors is 10000,  and the average price of a vehicle with four doors is close to 12000.  
# MAGIC With this plot, we have gained enough insights fromthe  data and our data is ready to build a model.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### There are ways to explore relationships between more than two variables; although it can get a bit more complicated to interpret.

# COMMAND ----------

# Create a pivot table for car manufactures and fuel with horsepower rate as values
grouped = pd.pivot_table(data=df,index='make',columns='fuel-type',values='horsepower',aggfunc='mean')

# Create a heatmap to visualize manufactures, fuel type and horse power
plt.figure(figsize=[12,10])
sns.heatmap(grouped, annot=True, cmap='coolwarm', center=0.117)

plt.title("Horse Power per Manufacturer")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## II - Feature Engineering and Selection

# COMMAND ----------

# MAGIC %md
# MAGIC This will be developed in the next modules

# COMMAND ----------

# MAGIC %md
# MAGIC ## III - Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC This will be developed in the next modules

# COMMAND ----------

# MAGIC %md
# MAGIC ## IV - Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC This will be developed in the next modules

# COMMAND ----------

# MAGIC %md
# MAGIC # Your Turn!

# COMMAND ----------

# MAGIC %md
# MAGIC #### Open the datasets available for the use cases and start the EDA.  
# MAGIC #### You will be able to make a better decision on which one to use and how to exploit them.

# COMMAND ----------

df = pd.read_csv("../../3_artificial_use_case/1_Classification_RECOMMENDED/Bank_Dataset/bank-additional-full.csv", sep=";")
df.head()

# COMMAND ----------

df = pd.read_csv("../../3_artificial_use_case/2_Regression_RECOMMENDED/Datasets/2015.csv")
df.head()

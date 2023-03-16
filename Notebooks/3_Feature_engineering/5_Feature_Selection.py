# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Loading libraries and Classes

# COMMAND ----------

# Import necessary libraries for this notebook
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from IPython.display import Image

# Import train_test_split to separate train and test set
from sklearn.model_selection import train_test_split
# Import VarianceThreshold to removes all low-variance features
from sklearn.feature_selection import VarianceThreshold

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC An **important** thing to remember is that we should perform feature selection in conjunction with the model selection

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Constant Features
# MAGIC Constant features do not provide any information useful for further analysis or predicting the target variable. These features provide only a single value for all of the observations in the dataset. Therefore, we can remove them from the dataset.
# MAGIC 
# MAGIC We will be working with the subset of Santander Bank dataset \\(^{1}\\) (30 000 rows), which contain anonymized features to predict customer satisfaction regarding their experience with the bank.

# COMMAND ----------

# Load the subset dataset called 'subset_santander.csv' and store it to variable data
data = pd.read_csv('Data/subset_santander.csv')

# Print the shape of the dataframe and get the first 10 rows
print(data.shape)
data.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Feature selection should be performed only on our training data to avoid overfitting. Let's split our dataset and drop our target feature 'TARGET'.

# COMMAND ----------

# Separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels = ['TARGET'], axis = 1),
                                                    data['TARGET'],
                                                    test_size = 0.3,
                                                    random_state = 42)

# Get the shape of training and testing set
X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC There are 370 features in our dataset. We can now look at whether there are some constant features in `X_train` set using the `.var()` method which computes the variance along the columns. Within this function we can specify argument `ddof = 1`. For more information see the [documentation](https://numpy.org/doc/stable/reference/generated/numpy.var.html).

# COMMAND ----------

# Get the features that have the variance equal to zero
# Optional: Specify ddof = 1 within the `.var()` function 

our_constant_features = X_train.loc[:, X_train.var(ddof = 1) == 0]

# COMMAND ----------

# Print our_constant_features
our_constant_features

# COMMAND ----------

# MAGIC %md
# MAGIC There are 64 features with zero variance, which will be removed from our dataset. 

# COMMAND ----------

# Remove constant features from X_train, do not forget specify argument inplace = True
X_train.drop(labels= our_constant_features, axis=1, inplace=True)
# Remove constant features from X_test, do not forget specify argument inplace = True
X_test.drop(labels= our_constant_features, axis=1, inplace=True)

# Get the shape after removing constant features
X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Quasi-constant Features
# MAGIC 
# MAGIC Quasi-constant features have very low variance (close to 0) and contain little information, which is not useful for us. These approximately constant features won't help the ML model's performance, therefore we should consider removing them. 
# MAGIC 
# MAGIC We could filter quasi-constant features with Pandas in a similar way as we did with constant features, with one difference - we would set a specific threshold. Nevertheless, now we'll leave Pandas behind and rather use scikit learn which offers a more convenient way to find quasi-constant features. 
# MAGIC 
# MAGIC In the `sklearn.feature_selection module` we can find a feature selector called `VarianceThreshold()`, which finds all features with low variance (based on a specified threshold) and removes them. You can find more information about this selector [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html).
# MAGIC 
# MAGIC As a first step, we define our selector for quasi-constant features with a threshold of 0.01. In other words, this is the minimum value of the variance we want to have in the dataset. 

# COMMAND ----------

# Define VarianceThreshold() object and specify parameter threshold = 0.01
our_selector = VarianceThreshold(threshold = 0.01)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we fit `our_selector` with the `X_train` data to find quasi-constant features.

# COMMAND ----------

# Fit X_train with our_selector
our_selector.fit(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Using `get_support()` method, we can get all of the features we want to keep along with their names. This _mask_ we will use later to assign names to columns.
# MAGIC 
# MAGIC *Note: You might wonder why we are saving the feature names in the `features_to_keep` variable. Scikit learn will always save the necessary state inside of the fitted transformer. However, in this example we only do this for our convience, so that we can later on go back from the nameless Numpy array to a nice dataframe with all the column names.*

# COMMAND ----------

# Get the mask of features we want to keep in the dataset
features_to_keep = X_train.columns[our_selector.get_support()]

# Print the length
print('The number of features that will be kept: {}'.format(len(features_to_keep)))

# COMMAND ----------

# MAGIC %md
# MAGIC The next step is to transform `X_train` and `X_test` using `our_selector`.

# COMMAND ----------

# Transform X_train and X_test = in this step, the quasi-constant featues will be finally removed
X_train = our_selector.transform(X_train)
X_test = our_selector.transform(X_test)

# Get the shape of X_train and X_test
X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC After the trnsformation `X_train` is a `numpy.ndarray` object and needs to be transformed into a Pandas DataFrame again. Here, we use our created `features_to_keep` variable to assign column names.

# COMMAND ----------

# Print X_train
X_train

# COMMAND ----------

# Convert X_train to a Pandas DataFrame
X_train= pd.DataFrame(X_train)
# Using the '.columns' attribute assign column names
X_train.columns = features_to_keep

# Convert X_test to a Pandas DataFrame
X_test= pd.DataFrame(X_test)
# Using the '.columns' attribute assign column names
X_test.columns = features_to_keep

# Get the first 5 rows of X_train
X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Duplicated Features (READ-ONLY)
# MAGIC Duplicated features are totally redundant features, thus not providing any useful or new information for improving the model's performance.
# MAGIC 
# MAGIC To better understand how duplicated features can be treated using Pandas we create new DataFrame. We've already seen the `duplicated()` function which returns a boolean Series denoting duplicate rows. To identify duplicated features, we have to first transpose our data frame, in other words, we swap the rows and columns. More information [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transpose.html).
# MAGIC 
# MAGIC You might wonder again, why are we not using the `scikit-learn`? The reason is that **duplicated features should be already addressed within data integration and preprocessing**. You might remember that these are the earliest stages. The reason is that duplicated values usually occur when we are merging data from various sources. It was not a priority for scikit developers to implement a specific transformer for this.
# MAGIC 
# MAGIC We are doing an ugly operation of swapping rows and columns to make use of Pandas functionality and make this operation as easy as possible. Yes, we are only advising you to do this with a small dataset. If you have a *big* dataset, for example counted in TBs, you should not and most likely will not be able to do this.

# COMMAND ----------

# Run this code to create new DataFrame
our_data  =[(1, 'Colin Trevorrow', 124, 150 ,'Colin Trevorrow', 150, 'Jurassic World'),
           (2, 'George Miller', 120, 55, 'George Miller', 55, 'Mad Max: Fury Road'),
           (3, 'Robert Schwentke', 119, 112, 'Robert Schwentke', 112, 'Insurgent'),
           (4, 'J.J. Abrams', 136, 220,'J.J. Abrams', 220, 'Star Wars: The Force Awakens'),
           (5, 'James Wan', 137, 154, 'James Wan', 154, 'Furious 7'),
           (6, 'Bruce Brown', 95, 25, 'Bruce Brown', 25, 'The Endless Summer'),
           (7, 'Woody Allen', 80, 15, 'Woody Allen', 15, 'What`s Up, Tiger Lily?'),
           (8, 'James Cameron', 162, 180, 'James Cameron', 180, 'Avatar'),
           (9, 'Carl Tibbetts', 74, 44, 'Carl Tibbetts', 44, 'Black Mirror: White Christmas'),
           (10, 'Harold P. Warren', 74, 8, 'Harold P. Warren', 8, 'Manos: The Hands of Fate')]

movies = pd.DataFrame(our_data, columns= ['id', 'director', 'runtime','total_votes', 'name', 'number_of_votes', 'title'])

# COMMAND ----------

# Print movies DataFrame
movies

# COMMAND ----------

# Print the shape of movies
movies.shape

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, the movies DataFrame contains 10 rows and 7 features. Now we use the `.transpose()` method.

# COMMAND ----------

# Transpose movies and store it to variable movies_transpose
movies_transpose = movies.transpose()
movies_transpose

# COMMAND ----------

# Get the shape of movies_transpose
movies_transpose.shape

# COMMAND ----------

# MAGIC %md
# MAGIC After transposing, there are 7 rows (features) and 10 columns in `movies_transpose`.
# MAGIC 
# MAGIC Now we apply chained `duplicated().sum()` function on `movies_transpose` that give us the total number of duplicated rows (features).

# COMMAND ----------

# Get the total number of duplicated rows (features)
movies_transpose.duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC There are 2 duplicated rows (features), containing the same observations. We can drop duplicated rows using `.drop_duplicates()`. By setting `keep = 'first'` parameter, we determine which duplicated row we want to keep.

# COMMAND ----------

# Drop duplicates and store the result in the variable unique_features
unique_features = movies_transpose.drop_duplicates(keep = 'first').transpose()

# COMMAND ----------

# Get duplicated features and store the result in the variable duplicated_feature
duplicated_features = [column for column in movies.columns if column not in unique_features]
duplicated_features

# COMMAND ----------

# Drop the duplicated features from the original DataFrame
movies.drop(labels = duplicated_features, axis = 1, inplace = True)
movies

# COMMAND ----------

# MAGIC %md
# MAGIC However, this approach is not computationally and memory-efficient if you have a really large DataFrame with thousands of rows. As `scikit learn` does not offer a method to handle duplicated features, we need to create some function for this purpose. Then we drop duplicated features using Pandas' `.drop()` method.

# COMMAND ----------

# Create an empty list for duplicated features
features_duplicates = []

# Create a for loop for iterating over the range of columns from the X_train set
for col in range(len(X_train.columns)):
    column_1 = X_train.columns[col]
    # Find duplicated features by comparing columns using .equals
    for column_2 in X_train.columns[col + 1:]:
        if X_train[column_1].equals(X_train[column_2]):
            features_duplicates.append(column_2)
            
len(features_duplicates)

# COMMAND ----------

# Drop duplicated features from X_train and X_test
X_train.drop(labels = features_duplicates, axis = 1, inplace = True)
X_test.drop( labels = features_duplicates, axis = 1, inplace = True)

# COMMAND ----------

# Get the shape of X_train and X_test
X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC `scikit learn` module offers many methods such as selecting features based on their importance but we will not go there. You can find these methods in the [documentation](https://scikit-learn.org/stable/modules/feature_selection.html). Now we'll look at the correlation between features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Correlation
# MAGIC Features with high correlation have almost the same effect on the target feature. We can visualize relationships between features using `.corr()` method to understand the data better.

# COMMAND ----------

# Find the correlation among the columns and store it in variable correlation_matrix
correlation_matrix = X_train.corr()

# COMMAND ----------

# Plot the correlation matrix 
plt.figure(figsize=(11,11))
sns.heatmap(correlation_matrix, cmap = 'Blues');

# COMMAND ----------

# MAGIC %md
# MAGIC We'll find the highly correlated features using a function based on correlation coefficients above the threshold of 0.8.

# COMMAND ----------

def correlation(dataset, threshold):
    # Create set for correlated columns
    corelated_cols = set()  
    # Compute correlation 
    corr_matrix = dataset.corr()
    for c in range(len(corr_matrix.columns)):
        for j in range(c):
            # Take absolute correlation coefficient value 
            # If abs values are above threshold ...
            if abs(corr_matrix.iloc[c, j]) > threshold: 
                # ... Get name of column
                colname = corr_matrix.columns[c]
                corelated_cols.add(colname)
    return corelated_cols

# COMMAND ----------

# Use correlation function on X_train with threshold 0.8
corr_features_to_drop = correlation(X_train, 0.8)
len(set(corr_features_to_drop))

# COMMAND ----------

# Drop correlated features from X_train and X_test
X_train.drop(labels = corr_features_to_drop, axis = 1, inplace = True)
X_test.drop(labels = corr_features_to_drop, axis = 1, inplace = True)

# COMMAND ----------

# Get the shape of X_train and X_test
X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## TASK
# MAGIC 
# MAGIC You will be using an altered dataset containing variants of the Portuguese 'Vinho Verde' wine \\(^{2}\\). The features provide information about wine samples recorded based on physicochemical tests. There is also the target feature that denotes the quality score of the sample. 

# COMMAND ----------

# Load the dataset 'wine_quality.csv' and store it to variable wine
wine = pd.read_csv('Data/quality_of_wine.csv', sep = ',', )
# Get the first 10 rows
wine.head(10)

# COMMAND ----------

# Print the dataframe's datatypes
wine.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC Several numerical features are stored as float or integer, and one feature is stored as a string in our dataset. 
# MAGIC 
# MAGIC These numerical variables can be used to predict the quality of the wine samples. So the **'quality'** column is our **target feature**.  

# COMMAND ----------

# Separate the dataset 'wine' into training and testing set
# Store it in variables: X_training, X_testing, y_training, y_testing
# Drop the target feature 'quality'
# Set test_size = 0.3 and random_state = 42

X_training, X_testing, y_training, y_testing = train_test_split(wine.drop(labels = ['quality'], axis=1), 
                                                                wine['quality'],
                                                                test_size = 0.3,
                                                                random_state = 42)

# Get the shape of training and testing set
X_training.shape, X_testing.shape

# COMMAND ----------

# MAGIC %md
# MAGIC As we already know there is one non-numerical variable ('type'). Let's look at the unique values of this feature.

# COMMAND ----------

# Print unique values of 'type' column in X_training and X_testing sets
print(X_training['type'].unique())
print(X_testing['type'].unique())

# COMMAND ----------

# MAGIC %md
# MAGIC The datasets can also contain this type of constant feature stored as a string and have only one unique value/category. As this variable is not really helpful, we will drop it from the dataset. 

# COMMAND ----------

# TASK >>>> Remove the constant feature from X_training using '.drop()'. Do not forget to specify the argument inplace = True

# TASK >>>> Remove the constant feature from X_testing using '.drop()'. Do not forget to specify the argument inplace = True

# Get the shape of X_training and X_testing sets

# COMMAND ----------

# MAGIC %md
# MAGIC Now we want to select only those features that have a variance above the threshold = 0.01. Again, we will find quasi-constant features using scikit learn's `VarianceThreshold` as we did in the previous example.

# COMMAND ----------

# TASK >>>> Define a VarianceThreshold() object, specify the parameter threshold = 0.01 and store it in variable 'selector'

# COMMAND ----------

# TASK >>>> Fit X_training with 'selector'

# COMMAND ----------

# Get a mask of those features we want to keep in the dataset and store it in the variable 'features_we_keep'
features_we_keep = X_training.columns[selector.get_support()]
# Print the length of the variable features_we_keep
print('The number of features that will be kept: {}'.format(len(features_we_keep)))

# COMMAND ----------

# Print the quasi-constant features that we are meant to drop using a for loop 
for column in X_training.columns:
    if column not in features_we_keep:
        print(column)

# COMMAND ----------

# TASK >>>> Transform X_training 

# TASK >>>> Transform X_testing

# Get the shape of X_training and X_testing

# COMMAND ----------

# Convert X_training to Pandas DataFrame
X_training = pd.DataFrame(X_training)
# Using the '.columns' attribute assign column names
X_training.columns = features_we_keep

# Convert X_testing to Pandas DataFrame
X_testing = pd.DataFrame(X_training)
# Using the '.columns' attribute assign column names
X_testing.columns = features_we_keep

# Get the first 10 rows of X_train
X_training.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Find whether our dataset contains duplicated features. You can copy-paste the `for` loop we've already used.

# COMMAND ----------

# Find duplicated features in X_training set
features_duplicates = []
for col in range(len(X_training.columns)):
     
    column_1 = X_training.columns[col]
    
    for column_2 in X_training.columns[col + 1:]:
        if X_training[column_1].equals(X_training[column_2]):
            features_duplicates.append(column_2)
            
len(features_duplicates)

# COMMAND ----------

# Print the features names
features_duplicates

# COMMAND ----------

# TASK >>>> Drop these duplicated features from X_training and X_testing

# COMMAND ----------

# Get the shape of X_training and X_testing
X_training.shape, X_testing.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # Apendix
# MAGIC 
# MAGIC Data sources:
# MAGIC 
# MAGIC \\(^{1}\\) Santander dataset: https://www.kaggle.com/c/santander-customer-satisfaction/data
# MAGIC 
# MAGIC \\(^{2}\\) Wine quality dataset: https://archive.ics.uci.edu/ml/datasets/wine+quality
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)

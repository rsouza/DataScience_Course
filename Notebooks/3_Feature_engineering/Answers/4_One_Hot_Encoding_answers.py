# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Loading libraries and Classes

# COMMAND ----------

# Import pandas and numpy libraries
import pandas as pd

import warnings
warnings.simplefilter("ignore")

from IPython.display import Image

# Import train_test_split to separate train and test set
from sklearn.model_selection import train_test_split
# Import OneHotEncoder for one hot encoding 
from sklearn.preprocessing import OneHotEncoder
# Import LabelEncoder for target feature encoding
from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why should we use Encoding ? 
# MAGIC 
# MAGIC As we already know, we can't throw the data right away into machine learning models. We need to treat them in a specific way so our model's algorithm can work with them. **Machine learning algorithm work with vectors of numbers**, so when it comes to values represented as a string there is an issue. `scikit learn`, an industry-standard library used for machine learning, does not accept categorical values represented as strings as well.
# MAGIC 
# MAGIC Imagine we have categorical variables stored as string in the dataset. For understanding what the encoding looks like, here's a simple example.

# COMMAND ----------

# Run this code
dataframe = pd.DataFrame({'id': range(8), 'amount': [15,85,17,22,56,84,15,48],
                          'color':['black','white','black','black','white','white','black','black'],
                          })
mapping = {'black': 1,
          'white':0}
# Mapping values
mapped_df = dataframe['color'].map(mapping)
# Comparison
map_dataframe = pd.concat([dataframe, mapped_df], axis = 1)
map_dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC The unique categories of the 'color' column have been converted into numerical form as 1 when the 'black' category is present and 0 otherwise. Of course, encoding categorical features using mapping or replacing can be very tedious and not effective if we have many categorical features and corresponding categories. Fortunately, you can find several encoding methods that serve for different encoding challenges.
# MAGIC 
# MAGIC -------
# MAGIC 
# MAGIC Categorical variables take only a limited number of possible values/categories and must be converted into numerical form. We should perform this conversion over the **training data** and propagate them to the unseen data (for example holdout data). 
# MAGIC 
# MAGIC **The main reason for this approach is that we do not know whether the future data will have all the categories present in the training data**. There could also be fewer or more categories. Therefore the encoders must learn patterns from the training data and use those learned categories in both training and testing sets.

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook we will use the Titanic and the Mushrooms datasets. 

# COMMAND ----------

# Load Titanic dataset using columns 'Survived','Sex','Embarked','Cabin' and store it in 'data'
data = pd.read_csv('../Data/titanic_data.csv', usecols = ['Survived','Sex','Embarked','Cabin'])
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC For this demonstration, let's capture only the first letter of 'Cabin'.

# COMMAND ----------

# Capture only first letter of Cabin using .str[0] 
data['Cabin'] = data['Cabin'].str[0]
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we split our data into training and testing set.

# COMMAND ----------

# Separate the DataFrame into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data[['Sex', 'Embarked','Cabin']],  
                                                    data['Survived'],  
                                                    test_size = 0.3,  
                                                    random_state = 42)
# Get the shape of training and testing set
X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cardinality of the categorical features
# MAGIC 
# MAGIC Let's explore how many unique values each of the categorical features has.

# COMMAND ----------

# Get the unique values of categorical features
for column in X_train.columns:
    print(column)
    print(X_train[column].unique())

# COMMAND ----------

# MAGIC %md
# MAGIC We'll look at the methods for encoding these categories and how these methods handle missing values present in the data.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 1. One-Hot Encoding with Pandas
# MAGIC 
# MAGIC We can use Pandas method `pd.get_dummies()` to encode the categorical features. In the real world this encoding method shouldn't be used in ML pipelines (computationally and memory ineffective). However, in the case of some simple data analysis you should be able to use it. We'll look at how it works and what its advantages and limitations are.

# COMMAND ----------

# Get the dummy variables of feature 'Sex' using pd.get_dummies() 
dummies = pd.get_dummies(X_train['Sex'])
dummies.head()

# COMMAND ----------

type(dummies)

# COMMAND ----------

# MAGIC %md
# MAGIC The main advantages are that `get_dummies()` returns a DataFrame and preserves feature names for dummy variables. Also, we can use this method even if our data contains missing values. 
# MAGIC 
# MAGIC In this example it has created one column for the female category and one column for the male category according to its presence. We can compare the created dummy variables to the original 'Sex' variable using concatenation to see what happened.

# COMMAND ----------

# Concat the original Series 'Sex' from X_train with created dummy variables Series
result = pd.concat([X_train['Sex'], pd.get_dummies(X_train['Sex'])], axis = 1)
result

# COMMAND ----------

# TASK 1 >>>> Get dummy variables for the column 'Embarked'
#             Concat the original 'Embarked' Series with the created dummy variables Series
#             Store it in the variable result_2

result_2 = pd.concat([X_train['Embarked'], pd.get_dummies(X_train['Embarked'])], axis = 1)
result_2

# COMMAND ----------

# MAGIC %md
# MAGIC **Encoding into *k*-1 dummy variables**
# MAGIC 
# MAGIC Categorical variables should be encoded by creating *k*-1 binary variables. What does it mean, and why should we use it? 
# MAGIC 
# MAGIC Here *k* represents the number of distinct categories. In the feature 'Sex' there are only two categories of sex: male or female, so *k* = 2. We only need to create one binary variable (*k*-1 = 1) and still have all the information contained in the original dataset. In other words, if the value is 0 in all the binary variables, then it must be 1 in the final (not present) binary variable.
# MAGIC For example, if we have the variable with 5 categories (*k* = 5), we would create 4 binary variables (*k* - 1 = 4). 
# MAGIC 
# MAGIC This approach helps to eliminate the redundancy of the information. 
# MAGIC 
# MAGIC To create *k*-1 dummy variables we specify parameter `drop_first = True` to drop the first binary variable.

# COMMAND ----------

dummies_2 = pd.get_dummies(X_train['Sex'], drop_first = True)
dummies_2

# COMMAND ----------

# MAGIC %md
# MAGIC If we create dummy variables for the entire dataset, the prefixes (variables names) will be generated automatically. It doesn't return only 'male', but also the variable's name.

# COMMAND ----------

# Get dummy variable for entire train set
dummy_data = pd.get_dummies(X_train, drop_first = True)
dummy_data

# COMMAND ----------

# TASK 2 >>>> Get dummy variables for the entire test set and store them in the variable dummy_data_2

dummy_data_2 = pd.get_dummies(X_test, drop_first = True)
dummy_data_2

# COMMAND ----------

# MAGIC %md
# MAGIC **KEY LEARNING** We can notice that training and testing sets have a different number of dummy variables. In the testing set there is no category _'Cabin T'_. Therefore dummy variables for this category cannot be created. As the training set and the testing set must be of the same shape, `scikit learn's` models won't accept these as inputs. **Our entire modeling pipeline can fail because of this! We did not save the "state" of how many dummies should leave this part.** The the pipeline fails, our model does not predict, money is lost, people scream in panic, senior engineers debug over night and protesters burn the cars in the streets! I think you get the point.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. One-Hot Encoding with Scikit-learn
# MAGIC 
# MAGIC The `sklearn.preprocessing` module offers the `OneHotEncoder()` class which encodes categorical features by creating binary columns for each unique category of variables using a one-hot encoding scheme. The output is not a DataFrame, but a NumPy array. You can find the documentation of `OneHotEncoder` [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
# MAGIC 
# MAGIC ----
# MAGIC Firstly we need to create the encoder object where we can specify a set of parameters.
# MAGIC Then we'll fit `OneHotEncoder` to the set `X_train`. There we first have to fill in missing values as `OneHotEncoder` doesn't except those. Using the `.categories_` attribute we'll find all of the determined categories. 

# COMMAND ----------

# MAGIC %md
# MAGIC Before we start with scikit, don't forget that we need to get rid of the missing values. Let's just replace them with a string "missing".

# COMMAND ----------

X_train = X_train.fillna('Missing')
X_test = X_test.fillna('Missing')

# COMMAND ----------

# MAGIC %md
# MAGIC Now we get to scikit. If you are confused over the word "sparse", don't worry. It is just a cool concept of how we can store a matrix in a more memory efficient way.

# COMMAND ----------

# Create the encoder
# Set parameter categories = 'auto' to determine the categories automatically from the training set
# Set parameter sparse = False to return a dense array 
# Set parameter handle_unknown = 'error' to raise an error if an unknown categorical feature is present during the transform
encoder = OneHotEncoder(categories='auto', sparse=False, handle_unknown='error')

#  Fit the encoder 
encoder.fit(X_train)

# COMMAND ----------

# We can inspect the categories used with the .categories_ attribute
encoder.categories_

# COMMAND ----------

# MAGIC %md
# MAGIC To transform `X_train` using our encoder, we need to fill in missing values again. Since the output will be a NumPy array, we'll have to convert it to a Pandas DataFrame. 

# COMMAND ----------

# Transform X_train using encoder 
training_set = encoder.transform(X_train)

# Convert X_train to a DataFrame
pd.DataFrame(training_set).head()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, after transforming the data the names of the features are not returned, which is inconvenient for feature exploration. There is a method for retrieving these names called `.get_feature_names()` which we can apply to the columns. Let's repeat the entire process of transforming.

# COMMAND ----------

# Transform X_train using one-hot encoding and return feature names
training_set = encoder.transform(X_train)
training_set = pd.DataFrame(training_set)
training_set.columns = encoder.get_feature_names()
training_set.head()

# COMMAND ----------

# TASK 2 >>>> Transform X_test using one-hot encoding in the same way as we did with X_train and store it in the variable testing_set
#             Inspect the first 5 rows to see the result

testing_set = encoder.transform(X_test)
testing_set = pd.DataFrame(training_set)
testing_set.columns = encoder.get_feature_names()
testing_set.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that after encoding the training set and testing set have the same number of features. 

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Encoding target variable
# MAGIC 
# MAGIC For encoding the target variable stored as a string datatype, we can use `LabelEncoder` class from the scikit learn module. `LabelEncoder` normalizes labels to have values between 0 and n_classes-1. You can find the documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder).
# MAGIC 
# MAGIC Let's look at the simple example of using this class on dog breeds. Firstly we create a `LabelEncoder` object and then we fit our data.

# COMMAND ----------

# Create LabelEncoder object
label_encoder = LabelEncoder()

# COMMAND ----------

# Fit data using label_encoder
label_encoder.fit(['Border Collie','Dachshund','Irish Setter','Papillon','Pug',
                   'Pembroke Welsh Corgi','Dachshund','Hokkaido','Pug'])

# COMMAND ----------

# After we fitted our data we can access the used categories
list(label_encoder.classes_)

# COMMAND ----------

# Transform the data
encoded_labels = label_encoder.transform(['Border Collie','Dachshund','Irish Setter','Papillon','Pug',
                                          'Pembroke Welsh Corgi','Dachshund','Hokkaido','Pug'])
encoded_labels

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of two binary values (0 and 1), we now have a sequence of numbers which are not in ascending order. The reason for this is that the numbering is assigned in alphabetical order.
# MAGIC 
# MAGIC -------
# MAGIC 
# MAGIC ### TASK
# MAGIC Now it's your turn to encode categorical variables in the **Mushrooms classification** dataset.

# COMMAND ----------

# Run this code to create a list of selected features
cols_to_use = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
               'stalk-shape', 'stalk-root', 'stalk-surface-above-ring']

# COMMAND ----------

# Load the dataset 'Data/mushrooms.csv' and store it in mushrooms
# Specify parameter usecols = cols_to_use
mushrooms = pd.read_csv('../Data/mushrooms.csv', usecols = cols_to_use)
# Get the first 5 rows
mushrooms.head()

# COMMAND ----------

# Get the unique values for all of the features in mushrooms that will be encoded
for column in mushrooms.columns:
    print(column)
    print(mushrooms[column].unique())

# COMMAND ----------

# MAGIC %md
# MAGIC You should see that one of the unique values there is '?' in the column 'stalk-root'. Replace this incorrectly stored value with 'Missing'.

# COMMAND ----------

# Use .replace() method to replace '?' with 'Missing'
mushrooms['stalk-root'] = mushrooms['stalk-root'].replace('?','Missing')

# COMMAND ----------

# Split mushrooms into training and testing set
# Set test_size = 0.3
# Set random_state = 42
X_train, X_test, y_train, y_test = train_test_split(mushrooms[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                                               'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                                                               'stalk-shape', 'stalk-root', 'stalk-surface-above-ring']], 
                                                    mushrooms['class'], 
                                                    test_size = 0.3, 
                                                    random_state = 42)

# Get the shape of X_train and X_test
X_train.shape, X_test.shape

# COMMAND ----------

# TASK >>>> Create a OneHotEncoder object where the categories will be automatically determined
# The result will be a dense array
# If an unknown categorical feature will be present during transform it will raise 'error'
# Store it in the variable encoder

encoder = OneHotEncoder(categories = 'auto',
                        sparse = False,
                        handle_unknown = 'error')

# COMMAND ----------

# TASK >>>> Fit X_train using encoder

encoder.fit(X_train)

# COMMAND ----------

# TASK >>>> Get the used categories

encoder.categories_

# COMMAND ----------

# TASK >>>> Transform X_train and convert it to a Pandas DataFrame
# You can assign it to X_train
# Get the feature names and inspect the changes after transforming

X_train = encoder.transform(X_train)
X_train = pd.DataFrame(X_train)
X_train.columns = encoder.get_feature_names()
X_train

# COMMAND ----------

# TASK >>>> Transform X_test and convert it to a Pandas DataFrame
# You can assign it to X_test
# Get the feature names and inspect the changes after transforming

X_test = encoder.transform(X_test)
X_test = pd.DataFrame(X_test)
X_test.columns = encoder.get_feature_names()
X_test

# COMMAND ----------

# MAGIC %md
# MAGIC Our target feature 'class' also needs to be encoded. To do so, use `LabelEncoder`.

# COMMAND ----------

# TASK >>>> Create LabelEncoder object and store it in variable labels_encoder

labels_encoder = LabelEncoder()

# COMMAND ----------

# TASK >>>> Fit y_train using labels_encoder

labels_encoder.fit(y_train)

# COMMAND ----------

# Print the used categories
labels_encoder.classes_

# COMMAND ----------

# TASK >>>> Transform the y_train data and assign to y_train

y_train = labels_encoder.transform(y_train)

# COMMAND ----------

# Print y_train
y_train

# COMMAND ----------

# TASK >>>> Fit and transform y_test data in the same way

labels_encoder.fit(y_test)
y_test = labels_encoder.transform(y_test)
y_test

# COMMAND ----------

# MAGIC %md
# MAGIC ### Appendix
# MAGIC 
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)

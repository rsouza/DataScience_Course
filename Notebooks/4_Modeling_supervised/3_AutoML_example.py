# Databricks notebook source
# MAGIC %md
# MAGIC ## AutoML Basic Concepts
# MAGIC 
# MAGIC [Source](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Titanic_Kaggle.ipynb)

# COMMAND ----------

#!pip install -U tpot

# COMMAND ----------

# Import required libraries
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np

# COMMAND ----------

# Load the data
titanic = pd.read_csv('data_titanic/train.csv')
titanic.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Exploration
# MAGIC Before we get going with TPOT, we start with some simple data exploration to understand our data set. 

# COMMAND ----------

titanic.groupby('Sex').Survived.value_counts()

# COMMAND ----------

titanic.groupby(['Pclass','Sex']).Survived.value_counts()

# COMMAND ----------

id = pd.crosstab([titanic.Pclass, titanic.Sex], titanic.Survived.astype(float))
id.div(id.sum(1).astype(float), 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Munging
# MAGIC The first and most important step in using TPOT on any data set is to rename the target class/response variable to 'class'.

# COMMAND ----------

titanic.rename(columns={'Survived': 'class'}, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC At present, TPOT requires all the data to be in numerical format. As we can see below, our data set has 5 categorical variables which contain non-numerical values: 'Name', 'Sex', 'Ticket', 'Cabin' and 'Embarked'.

# COMMAND ----------

titanic.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC We then check the number of distinct levels that each of the five categorical variables have.

# COMMAND ----------

for cat in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, titanic[cat].unique().size))

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see, 'Sex' and 'Embarked' have only very few levels. Let's find out what they are.

# COMMAND ----------

for cat in ['Sex', 'Embarked']:
    print("Levels for catgeory '{0}': {1}".format(cat, titanic[cat].unique()))

# COMMAND ----------

# MAGIC %md
# MAGIC We then code these levels manually into numerical values. For NaN i.e. the missing values, we simply replace them with a placeholder value (-999). In fact, we perform this replacement for the entire data set.

# COMMAND ----------

titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})

# COMMAND ----------

titanic = titanic.fillna(-999)
pd.isnull(titanic).any()

# COMMAND ----------

# MAGIC %md
# MAGIC Since 'Name' and 'Ticket' have so many different levels, we drop them in this example from our analysis for the sake of simplicity. For 'Cabin', we encode the levels as digits using Scikit-learn's `MultiLabelBinarizer` and treat them as new features.

# COMMAND ----------

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])

# COMMAND ----------

CabinTrans

# COMMAND ----------

# MAGIC %md
# MAGIC Drop the unused features from the dataset.

# COMMAND ----------

titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)

# COMMAND ----------

assert (len(titanic['Cabin'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done

# COMMAND ----------

# MAGIC %md
# MAGIC We then add the encoded features to form the final dataset to be used with TPOT.

# COMMAND ----------

titanic_new = np.hstack((titanic_new.values,CabinTrans))

# COMMAND ----------

np.isnan(titanic_new).any()

# COMMAND ----------

# MAGIC %md
# MAGIC Keep in mind that the final data set is in the form of a numpy array. We can check the number of features in the final data set as follows.

# COMMAND ----------

titanic_new[0].size

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we store the class labels which we need to predicted in a separate variable.

# COMMAND ----------

titanic_class = titanic['class'].values

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Analysis using TPOT
# MAGIC To begin our analysis, we need to divide our training data into training and validation sets. The validation set is just to give us an idea of the test set error. The model selection and tuning is entirely taken care of by TPOT, so if we want to, we can skip the creation of this validation set.

# COMMAND ----------

training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.75, test_size=0.25)
training_indices.size, validation_indices.size

# COMMAND ----------

# MAGIC %md
# MAGIC After that, we proceed with calling the fit, score and export functions on our training dataset. To get a better idea of how these functions work, refer to the TPOT documentation here.
# MAGIC 
# MAGIC An important TPOT parameter to set is the **number of generations**. Since our aim is to just illustrate the use of TPOT, we set maximum optimization time to 2 minutes (`max_time_mins=2`). On a standard laptop with 4GB RAM it takes roughly 5 minutes per generation to run. For each added generation, it should take 5 minutes more. Thus, for the default value of 100, the total run time could be roughly around 8 hours.

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)  # As MLFlow is not totally integrated with TPOT, we disable the autologging when running on Databricks

# COMMAND ----------

tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=40)

# COMMAND ----------

tpot.fit(titanic_new[training_indices], titanic_class[training_indices])

# COMMAND ----------

tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values)

# COMMAND ----------

tpot.export('/tmp/tpot_titanic_pipeline.py')

# COMMAND ----------

# MAGIC %md
# MAGIC Let's have a look at the generated code. As we can see, the random forest classifier performed the best on the given dataset out of all the other models that TPOT currently evaluates on. If we ran TPOT for more generations, then the score should improve further.
# MAGIC 
# MAGIC Run:
# MAGIC > `%load tpot_titanic_pipeline.py`

# COMMAND ----------

#%load /tmp/tpot_titanic_pipeline.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8308382897542362
exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.8500000000000001, min_samples_leaf=18, min_samples_split=14, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Make predictions on the submission data

# COMMAND ----------

# Read in the submission dataset
titanic_sub = pd.read_csv('data_titanic/test.csv')
titanic_sub.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC When looking at fresh data to make predictions on, the most important step is to **check for new levels in the categorical variables** of the submission data set which were absent in the training set. We identify them and set them to our placeholder value of `-999`, i.e., we treat them as missing values. This ensures training consistency, as otherwise the model would not know what to do with the new levels in the submission data set.

# COMMAND ----------

for var in ['Cabin']: #,'Name','Ticket']:
    new = list(set(titanic_sub[var]) - set(titanic[var]))
    titanic_sub.loc[titanic_sub[var].isin(new), var] = -999

# COMMAND ----------

# MAGIC %md
# MAGIC We then carry out the data munging steps as done earlier for the training dataset.

# COMMAND ----------

titanic_sub['Sex'] = titanic_sub['Sex'].map({'male':0,'female':1})
titanic_sub['Embarked'] = titanic_sub['Embarked'].map({'S':0,'C':1,'Q':2})

# COMMAND ----------

titanic_sub = titanic_sub.fillna(-999)
pd.isnull(titanic_sub).any()

# COMMAND ----------

# MAGIC %md
# MAGIC While calling `MultiLabelBinarizer` on the submission data set, we first fit on the training set again to learn the levels and then transform the submission data set values. This further ensures that only those levels that were present in the training data set are transformed. If new levels are still found in the submission data set then it will return an error and we need to go back and check our earlier step of replacing new levels with the placeholder value.

# COMMAND ----------

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
SubCabinTrans = mlb.fit([{str(val)} for val in titanic['Cabin'].values]).transform([{str(val)} for val in titanic_sub['Cabin'].values])
titanic_sub = titanic_sub.drop(['Name','Ticket','Cabin'], axis=1)

# COMMAND ----------

# Form the new submission data set
titanic_sub_new = np.hstack((titanic_sub.values,SubCabinTrans))

# COMMAND ----------

np.any(np.isnan(titanic_sub_new))

# COMMAND ----------

# Ensure an equal number of features in both the final training and submission dataset
assert (titanic_new.shape[1] == titanic_sub_new.shape[1]), "Not Equal" 

# COMMAND ----------

# Generate the predictions
submission = tpot.predict(titanic_sub_new)

# COMMAND ----------

# Create the submission file
final = pd.DataFrame({'PassengerId': titanic_sub['PassengerId'], 'Survived': submission})
#final.to_csv('submission.csv', index = False)

# COMMAND ----------

final.shape

# COMMAND ----------

# MAGIC %md
# MAGIC There we go! We have successfully generated the predictions for the 418 data points in the submission dataset, and we're good to go ahead to submit these predictions on Kaggle.

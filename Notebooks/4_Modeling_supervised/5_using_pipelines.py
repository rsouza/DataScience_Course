# Databricks notebook source
# MAGIC %md
# MAGIC # Nice Pipeline
# MAGIC 
# MAGIC In this notebook we present a nice example of a pipeline which we can use for training purposes. At first glance it looks messy and hard to read.  
# MAGIC But if you take a moment to really understand it you will notice the beauty of it!

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

# COMMAND ----------

# MAGIC %md
# MAGIC We need to import some transformers which are going to be inside of the pipeline.  
# MAGIC **This is not operational code, just an example of longer pipelines.**

# COMMAND ----------

#Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

#Dimensionality reduction
from sklearn.decomposition import NMF

#Imputation
from sklearn.impute import SimpleImputer

#Modeling
from sklearn.ensemble import RandomForestClassifier

#Other
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC The pipeline below will be a bit overwelming at first. Because of this we recommend the following steps to ease comprehension:
# MAGIC 
# MAGIC ## Step 1: Take a quick glance
# MAGIC Please take a quick look at the pipeline below and come back here.
# MAGIC 
# MAGIC ## Step 2: Slow walkthrough
# MAGIC Get a **high level view** of the pipeline:
# MAGIC - Look at the top. There is a `FeatureUnion` object, which is a wrapper for the entire feature engineering process.
# MAGIC - Look at the bottom. There is a `RandomForestClassifier` object, which is our predictive model.
# MAGIC 
# MAGIC Now we can go deeper into the `FeatureUnion` object we have instantiated, which is where the **feature engineering** is happening.
# MAGIC - `FeatureUntion` splits into three parts, depending on which features we are attempting to process:
# MAGIC     - On top we have numerical features.
# MAGIC     - In the middle we have categorical features.
# MAGIC     - At the bottom we have textual features.
# MAGIC - Now zoom out again and realize that this is wrapped under `FeatureUnion`, which means that **these features will be transformed in a parallel way and appended next to each other**.
# MAGIC 
# MAGIC ## Step 3: Zooming in
# MAGIC Only now let's **zoom into one part of our feature engineering**, for example into `numerical_features`, on the top:
# MAGIC - Inside of it, we right away use `ColumnTransformer` as we want to specify for which columns certain transformation will be applied to based on name or by type.
# MAGIC - Now we can already apply the transformers. But remember that `ColumnTransformer` by default drops all untransformed columns, which would mean that if we want to apply some transformations sequentially we would not be able to.
# MAGIC 
# MAGIC ## Step 4: Indentation
# MAGIC Finally, **get used to the indentation** (the whitespacing). Your code editor helps with this. Get used to this by clicking just behind the last visible character on the line where you are. For example go behing the last bracket on the line of `SimpleImputer`. Now if you hit Enter, it will land where a code should continue on the next line it you still want to stay within the element, which is a `Pipeline` object.

# COMMAND ----------

# MAGIC %md
# MAGIC Source 1: https://www.codementor.io/@bruce3557/beautiful-machine-learning-pipeline-with-scikit-learn-uiqapbxuj   
# MAGIC Source 2: http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

# COMMAND ----------

model_pipeline = Pipeline(steps=[
    ("features", FeatureUnion([
        ("numerical_features",
         ColumnTransformer([
             ("numerical",
              Pipeline(steps=[(
                  "impute_stage",
                  SimpleImputer(missing_values=np.nan, strategy="median")
              )]),
              ["feature_1"]
             )
         ])
        ), 
        ("categorical_features",
            ColumnTransformer([
                ("country_encoding",
                 Pipeline(steps=[
                     ("ohe", OneHotEncoder(handle_unknown="ignore")),
                     ("reduction", NMF(n_components=8)),
                 ]),
                 ["country"],
                ),
            ])
        ), 
        ("text_features",
         ColumnTransformer([
             ("title_vec",
              Pipeline(steps=[
                  ("tfidf", TfidfVectorizer()),
                  ("reduction", NMF(n_components=50)),
              ]),
              "title"
             )
         ])
        )
    ])
    ),
    ("classifiers", RandomForestClassifier())
])

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have created a pipeline and know how it is structured, we can work with easily:

# COMMAND ----------

model_pipeline.fit(train_data, train_labels.values)
predictions = model_pipeline.predict(predict_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. How to write that?
# MAGIC Alright, we are now slightly more comfortable with understanding how pipelines work. But how do we write them ourselves?   
# MAGIC The answer is: **from the outside inwards**. 
# MAGIC 
# MAGIC Let's walk through an example. Of course you can write things differently.    
# MAGIC At first, lay out a simple structure which separates your feature engineering (inside of `FeatureUnion`) and your predictive model:

# COMMAND ----------

# model_pipeline = Pipeline(steps=[
#     ("features", FeatureUnion([#all feature engineering goes here])),
#     ("classifiers", RandomForestClassifier())
# ])

# COMMAND ----------

# MAGIC %md
# MAGIC Secondly, depending on your features, split the feature engineering into various parts:

# COMMAND ----------

model_pipeline = Pipeline(steps=[
    ("features", FeatureUnion([("numerical_features", #numerical transformations), 
                               ("categorical_features", #categorical transformations), 
                               ("text_features", #textual transformations)
                              ])
    ),
    ("classifiers", RandomForestClassifier())
])

# COMMAND ----------

# MAGIC %md
# MAGIC Now you want to insert a `ColumnTransformer` as the transformations will be applied only to specific columns.

# COMMAND ----------

model_pipeline = Pipeline(steps=[
    ("features", FeatureUnion([("numerical_features", ColumnTransformer([#numerical transformations])),
                               ("categorical_features", ColumnTransformer([#categorical transformations])),
                               ("text_features", ColumnTransformer([#textual transformations]))
                              ])
    ),
    ("classifiers", RandomForestClassifier())
])

# COMMAND ----------

# MAGIC %md
# MAGIC You can put a `Pipeline` inside of the feature engineering, for example, in case you have transformers which need to be applied sequentially (such as numeric scaling and feature selection).  
# MAGIC 
# MAGIC At this point you can start inserting your individual transformations from before into the pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Reflect
# MAGIC Continue with this point only once you went through the pipeline above.  
# MAGIC 
# MAGIC Usually we think that nicely written code costs significantly more effort than code scraped together in whichever way. Now that we went through the composite estimators properly, you know that it might be even simpler in many cases, not to mention more robust.  
# MAGIC 
# MAGIC At this point, you are hopefully able to tell apart two things:  
# MAGIC - Data preprocessing and wrangling.
# MAGIC - Data preparation for ML (Feature Engineering)  
# MAGIC 
# MAGIC Always try to separate these things in your use case (code). That is why we present these topics separatedely. It will be of tremendous help in the long run to write code in this way.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Working Example  
# MAGIC [Source](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# COMMAND ----------

train = pd.read_csv("data_titanic/train.csv")
train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We will use ``ColumnTransformer`` where we select the columns by their names.   
# MAGIC We will then train our classifier with the following features:
# MAGIC 
# MAGIC **Numeric Features**
# MAGIC 
# MAGIC * 'Age': float;
# MAGIC * 'Fare': float.
# MAGIC 
# MAGIC **Categorical Features**
# MAGIC 
# MAGIC * 'Embarked': categories encoded as strings: ``{'C', 'S', 'Q'}``;
# MAGIC * 'Sex': categories encoded as strings: ``{'female', 'male'}``;
# MAGIC * 'Pclass': ordinal integers: ``{1, 2, 3}``.
# MAGIC 
# MAGIC We first create the preprocessing pipelines for both numeric and categorical data.
# MAGIC Note that 'Pclass' could either be treated as a categorical or a numeric
# MAGIC feature.

# COMMAND ----------

X = train.drop('Survived', axis=1)
y = train['Survived']

# COMMAND ----------

numeric_features = ["Age", "Fare"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), 
                                      ("scaler", StandardScaler())]
                              )

categorical_features = ["Embarked", "Sex", "Pclass"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features),
                                              ]
                                )

# COMMAND ----------

# MAGIC %md
# MAGIC Append a classifier, in this case a logistic regression, to preprocessing pipeline to arrive at a full prediction pipeline.

# COMMAND ----------

clf = Pipeline(steps=[("preprocessor", preprocessor), 
                      ("classifier", LogisticRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf.fit(X_train, y_train)

print("model score: %.3f" % clf.score(X_test, y_test))

# COMMAND ----------

clf

# COMMAND ----------

# MAGIC %md
# MAGIC **We can also select columns by data types when using `ColumnTransformer`.**   
# MAGIC When dealing with a cleaned dataset, the preprocessing can be automatic by
# MAGIC using the data types of the column to decide whether to treat a column as a numerical or categorical feature.
# MAGIC 
# MAGIC The function `sklearn.compose.make_column_selector` gives this possibility.
# MAGIC 
# MAGIC In practice, you will have to handle the column data type yourself.
# MAGIC If you want some columns to be considered as `category`, you will have to convert them into categorical columns. If you are using pandas, you can refer to their documentation regarding [Categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html).</p>
# MAGIC 
# MAGIC 
# MAGIC + First, we will transform the object columns into categorical.  
# MAGIC + Then we will only select a subset of columns to simplify our example.

# COMMAND ----------

X["Embarked"] = X["Embarked"].astype("category")
X["Sex"] = X["Sex"].astype("category")

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
subset_feature = ["Embarked", "Sex", "Pclass", "Age", "Fare"]
X_train, X_test = X_train[subset_feature], X_test[subset_feature]

# COMMAND ----------

X_train.info()

# COMMAND ----------

# MAGIC %md
# MAGIC We can observe that the 'Embarked' and 'Sex' columns were tagged as `category` columns.  
# MAGIC Therefore, we can use this information to dispatch the categorical columns to the ``categorical_transformer`` and the remaining columns to the ``numerical_transformer``.

# COMMAND ----------

from sklearn.compose import make_column_selector as selector

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
clf

# COMMAND ----------

# MAGIC %md
# MAGIC The resulting score is not exactly the same as the one from the previous pipeline because the dtype-based selector treats the 'Pclass' column as a numeric feature instead of a categorical feature:

# COMMAND ----------

selector(dtype_exclude="category")(X_train)

# COMMAND ----------

selector(dtype_include="category")(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC **Using the prediction pipeline in a grid search**   
# MAGIC 
# MAGIC A grid search can also be performed on the different preprocessing steps which make up the ``ColumnTransformer`` object. So you can optimize the classifier's hyperparameters as part of the pipeline.  
# MAGIC We will search for both the imputer strategy of the numeric preprocessing and the regularization parameter of the logistic regression using
# MAGIC `sklearn.model_selection.GridSearchCV`.

# COMMAND ----------

param_grid = {"preprocessor__num__imputer__strategy": ["mean", "median"],
              "classifier__C": [0.1, 1.0, 10, 100],
             }

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search

# COMMAND ----------

# MAGIC %md
# MAGIC Calling `fit` triggers the cross-validated search for the best hyper-parameters combination:

# COMMAND ----------

grid_search.fit(X_train, y_train)

print("Best params:")
print(grid_search.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC The internal cross-validation score obtained by those parameters is:

# COMMAND ----------

print(f"Internal CV score: {grid_search.best_score_:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC We can also extract the top grid search results as a pandas dataframe:

# COMMAND ----------

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results[["mean_test_score",
            "std_test_score",
            "param_preprocessor__num__imputer__strategy",
            "param_classifier__C",
           ]].head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC The best hyper-parameters have to be used to re-fit the final model on the full training set.  
# MAGIC We can evaluate that final model on held out test data that was not used for hyperparameter tuning.

# COMMAND ----------

print(f"best logistic regression from grid search: {grid_search.score(X_test, y_test):.3f}")

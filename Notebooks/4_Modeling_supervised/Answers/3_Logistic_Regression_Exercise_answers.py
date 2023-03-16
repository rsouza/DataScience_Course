# Databricks notebook source
# MAGIC %md
# MAGIC # Logistic regression

# COMMAND ----------

# Importing all the necessary libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics # to calculate accuracy measure and confusion matrix
import matplotlib.pyplot as plt 
import random
plt.rcParams["figure.figsize"] = (15,6)

# COMMAND ----------

# MAGIC %md
# MAGIC # Binary regression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dataset for binary regression

# COMMAND ----------

X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
print(datasets.load_breast_cancer().DESCR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make the data imbalanced
# MAGIC 
# MAGIC For the purpose of this exercise we will make the data imbalanced by removing 80% of the cases where `y==1`.

# COMMAND ----------

data = pd.concat([X,y], axis=1) # join X and y
data_neg = data.loc[data.target==0,:] # select only rows with negative target 
data_pos = data.loc[data.target==1,:].sample(frac=0.07, random_state=42) # select 7% of rows with positive target

data_imb = pd.concat([data_neg, data_pos]) # concatenate 7% of positive cases and all negative ones to have imbalanced data
X_imb = data_imb.drop(columns=['target'])
y_imb = data_imb.target
plt.title('frequency of the target variable')
plt.xlabel('target value')
plt.ylabel('count')
plt.hist(y_imb);


# COMMAND ----------

# MAGIC %md
# MAGIC Split into train and test sets.

# COMMAND ----------

#Task:

X_train , X_test , y_train , y_test = train_test_split(X_imb, y_imb, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC Fit the default `LogisticRegression()` to `X_train` and `y_train`.

# COMMAND ----------

#Task:

lr = LogisticRegression()
lr.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC The model failed to converge due to low number of iterations of the optimization solver. There are multiple solvers that can be chosen as a hyperparameter of the model. These also depend on the strategy that is chosen for regularization and for the multiclass problem. A description of which solver suits which problem is in the documentation. We have 3 options now:
# MAGIC 
# MAGIC - Increase the number of iterations until the default solver converges.
# MAGIC - Select a different optimization algorithm with a hyperparameter solver.
# MAGIC - Scale the input data which usually helps optimization algorithms to converge. However, if you do not use regularization, the scaling is not required for a logistic regression. It only helps with convergence.
# MAGIC 
# MAGIC ### Exercise
# MAGIC We will go with the last option. 
# MAGIC 
# MAGIC - Scale the data with a `StandardScaler()`.
# MAGIC - Fit and transform `X_train` and save it to `X_train_scaled`.
# MAGIC - Transform `X_test` and save it to `X_test_scaled`.

# COMMAND ----------

#Task:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - Fit the logistic regression to the scaled data.
# MAGIC - Predict on `X_train_scaled` and save the values to `y_hat`.
# MAGIC - What are the values that are returned from the `predict()` method?

# COMMAND ----------

#Task:

lr.fit(X_train_scaled, y_train)
y_hat = lr.predict(X_train_scaled)
y_hat

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC Print different metrics from `sklearn.metrics` for the predictions on the train set:
# MAGIC  - accuracy
# MAGIC  - confusion matrix
# MAGIC  - classification report

# COMMAND ----------

#Task:

print(f'accuracy {metrics.accuracy_score(y_train, y_hat)}')
print(f'confusion matrix\n {metrics.confusion_matrix(y_train, y_hat)}')
print(f'classification report\n {metrics.classification_report(y_train, y_hat)}')

# COMMAND ----------

# MAGIC %md
# MAGIC __WARNING__: You should never optimize for the results of the test set. The test set should be always set aside and you should evaluate only once you have decided on the final model. You will learn later in the course how to treat such situations in the lecture about hyperparameter tuning.
# MAGIC 
# MAGIC You can see from the confusion matrix that there are only 19 cases of the positive class in the train set while 2 of them were classified incorrectly and 17 correctly. We would rather want to predict correctly all those cases where `target = 1`. It is not a big deal if we tell the patient that she/he has a cancer while actually there is no cancer. The bigger problem is if we predict that the patient does not have a cancer while she/he actually has it. We can achieve this by changing the value of the threshold which by default is 50%. We should therefore lower the threshold for the probability.
# MAGIC 
# MAGIC After calling `.predict()` on your model it returned the predicted classes. Instead of predicting classes directly you can return probabilites for each instance using the `predict_proba()` method of the logistic regression model. One row is one observation. The first column is the probability that the instance belongs to the first class and the second column tells you about the probability of the instance belonging to the second class. Sum of the first and second column for each instance is equal to 1. You can find out which class is the first and which class is the second using the `classes_` attribute of the model. 
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - Return the classes with the `classes_` attribute.
# MAGIC - Return the probabilites of `X_train_scaled` with the `predict_proba()` method.
# MAGIC - Save the probabilities of the positive class in the variable `probs_train`.

# COMMAND ----------

#Task:

print(lr.classes_)
print(lr.predict_proba(X_train_scaled))
probs_train = lr.predict_proba(X_train_scaled)[:,1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 
# MAGIC 
# MAGIC Set the value of the threshold to 20% and use the probabilities saved in the variable `probs_train`: If the value of the probability is greater than the threshold then the prediction should be equal to 1. 
# MAGIC Hint: boolean values can be converted to 0/1 with `boolean_values.astype(int)`.
# MAGIC 
# MAGIC Return a confusion matrix using `.confusion_matrix()` as well as a classification report using `.classification_report()` for the train set.

# COMMAND ----------

#Task:

threshold = 0.2
preds_train = (probs_train>=threshold).astype(int)
print(metrics.confusion_matrix(y_train, preds_train))
print(metrics.classification_report(y_train, preds_train))

# COMMAND ----------

# MAGIC %md
# MAGIC It seems now that all the positive cases are classified correctly thanks to the change of the prediction threshold. Let's check the performance on the test data.
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - Save the probabilities of the positive class from the model on the `X_test_scaled` dataset in the variable `probs_test`.
# MAGIC - Convert the probabilities into predictions with a threshold 20% as above.
# MAGIC - Return a confusion matrix using `.confusion_matrix()` as well as a classification report using `.classification_report()` for the test set.

# COMMAND ----------

#Task:

probs_test = lr.predict_proba(X_test_scaled)[:,1]
preds_test=(probs_test>=0.2).astype(int)
print(metrics.confusion_matrix(y_test, preds_test))
print(metrics.classification_report(y_test, preds_test))

# COMMAND ----------

# MAGIC %md
# MAGIC Great. The model classifies all the 6 positive cases correctly on the test set. There are 2 cases when the patient did not have a cancer but the model predicted a cancer. 
# MAGIC What we were trying to optimize here is the **recall for a positive class** as we want to catch as many positive cases as possible. You can see the values of recall for the class 1 as a function of the threshold on the chart below.

# COMMAND ----------

recalls = []
for threshold in np.linspace(0,1,100):
    preds_train = (probs_train>=threshold).astype(int)
    recalls.append(metrics.classification_report(y_train, preds_train, output_dict=True,zero_division=1)['1']['recall'])
plt.xlabel('threshold')
plt.ylabel('recall for class 1')
plt.title("A search for optimal threshold")
plt.plot(np.linspace(0,1,100), recalls)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can return the parameters of the fitted model. This is convenient for automatic retraining of the model where you can extract the parameters of the best model and also set the parameters of the model with `set_params(**params)`.

# COMMAND ----------

lr.get_params()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regularization
# MAGIC 
# MAGIC Similarly to linear regression you can apply any of the l1, l2 and elastic net regularization techniques. Here the strength of the regularization is defined by the parameter C which is the inverse of alpha. This means that the smaller the C the stronger the regularization. The default value for C is 1.
# MAGIC 
# MAGIC Different regularization techniques work only with certain solvers, e.g. for the L1 penalty we have to use either liblinear or saga solver, L2 can be handled with newton-cg, lbfgs and sag solvers, elasticnet works only with saga solver. For elasticnet you can adjust the parameter `l1_ratio`.
# MAGIC 
# MAGIC ### Exercise
# MAGIC 
# MAGIC - Fit the logistic regression on `X_train_scaled` with a regularization of your choice through the parameter `penalty`.
# MAGIC - Change the solver if needed, see documentation.
# MAGIC - Try different values of C to see the effect on results. Try also stronger values such as 0.1, 0.01, ...
# MAGIC - Predict on `X_test_scaled` and return a classification report.

# COMMAND ----------

#Task:

lr = LogisticRegression(penalty='l1', C = 0.1, solver='liblinear')
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
print(metrics.classification_report(y_test, y_pred))

# COMMAND ----------

print(f'coefficients of the logistic regression:\n {lr.coef_}')

# COMMAND ----------

# MAGIC %md
# MAGIC If you fitted, for example, LogisticRegression(penalty='l1', C = 0.1, solver='liblinear') you would see that many of the coefficients are equal to 0. This behavior of l1 is expected not only for linear but also for logistic regression.

# COMMAND ----------

# MAGIC %md
# MAGIC # Multinomial Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC 
# MAGIC Here we will use here a dataset of handwritten numbers in a low resolution of 8x8 pixels. One picture is 64 values of pixels. There are 10 classes. You can see a few examples of these obserations in the picture below. We also perform the usual train test split and a scaling of features to help the optimizers converge.

# COMMAND ----------

data = datasets.load_digits()
X, y = data.data, data.target
X_train , X_test , y_train , y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for i in range(10):
    plt.subplot(2,5,i+1)
    num = random.randint(0, len(data))
    plt.imshow(data.images[num], cmap=plt.cm.gray, vmax=16, interpolation='nearest')

# COMMAND ----------

print(data.DESCR)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - Fit a default logistic regression on `X_train_scaled` and `y_train`.
# MAGIC - Predict and print the classification report on `X_test_scaled`.

# COMMAND ----------

#Task:

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_hat = lr.predict(X_test_scaled)

print(metrics.classification_report(y_test, y_hat)) # zero_division=1

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that in the classification report there is 1 row per class with all the statistics.
# MAGIC 
# MAGIC If you return probabilites with the `predict_proba()` method you will see that it has 1 column per class. It is a generalization of the binary case. The sum of all the probabilities per row is equal to 1.

# COMMAND ----------

probs = lr.predict_proba(X_test_scaled)
print(f'predict_proba shape: {probs.shape}')

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic regression can handle multinomial regression without any special setting. There is however a parameter that lets you choose the strategy for the multinomial problem which then is either _one vs rest_ or _softmax regression_. The choice of the strategy is also dependent on the selected solver. I.e. if `solver = 'liblinear'` then a softmax regression is not possible. In this case and if the problem is binary, the default strategy for `multi_class` is one vs rest. Otherwise it is softmax.
# MAGIC 
# MAGIC ### Exercise
# MAGIC - Fit a logistic regression to `X_train_scaled` and `y_train`. Use the parameter `multi_class` with the value 'ovr' which is the one vs rest strategy.
# MAGIC - Return the probabilities.

# COMMAND ----------

#Task:

lr = LogisticRegression(multi_class='ovr')
lr.fit(X_train_scaled, y_train)
y_hat = lr.predict(X_test_scaled)
probs = lr.predict_proba(X_test_scaled)
print(f'predict_proba shape: {probs.shape}')
np.sum(probs,axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)

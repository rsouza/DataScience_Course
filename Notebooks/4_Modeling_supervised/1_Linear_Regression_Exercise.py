# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams["figure.figsize"] = (15,4) # the charts will have a size of width = 15, height = 4


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import datasets

#%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC # Linear regression with 1 feature

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate a dataset for Linear Regression
# MAGIC For this lesson we will create an artificial dataset using the `sklearn.datasets` module of sklearn. 

# COMMAND ----------

# MAGIC %md
# MAGIC - With the `make_regression()` function of `datasets` we can generate a synthetical dataset for a regression problem.
# MAGIC - Here we generate 100 observations with 1 explanatory variable and a standard deviation for the gaussian noise of 40.
# MAGIC - If you want to read the documentation you can always **run the function name with a questionmark before the name** like in the cell below. This will open the documentation directly in jupyter notebok. You can also read the documentation on the 'internet, e.g. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

# COMMAND ----------

? datasets.make_regression # read the documentation

# COMMAND ----------

x, y, coeff = datasets.make_regression(n_samples = 100, 
                                       n_features = 1,
                                       noise = 40,
                                       coef = True,
                                       bias = 50,
                                       random_state = 42)

# Funny Note
# if you have read Hitchiker's Guide to the Galaxy then
# you know that 42 is the universal answer to life, the universe and everything
# https://www.quora.com/Why-do-we-choose-random-state-as-42-very-often-during-training-a-machine-learning-model

# COMMAND ----------

# MAGIC %md
# MAGIC We can plot x against y to see what the data look like.

# COMMAND ----------

#plt.figure(figsize=(15,4))
plt.scatter(x,y)
plt.title('Data')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Run the user defined function below which plots the observations and a line, and calculates the RMSE. You will use this function in the exercises!
# MAGIC It takes as inputs x and y, the intercept value and a coefficient value.

# COMMAND ----------

def plot_regression(x, y, bias, coeff):
    """
    The function plots a scatterpot of x, y and a line with bias and coefficient. It also calculates the RMSE.
    ---------------
    params:
    - x: points on the x-axis
    - y: points on the y-axis
    - bias: intercept of the line
    - coeff: slope of the line
     """
    y_hat = bias + x * coeff # predictions of x can be calculated easily 
                             # by multiplying the features with coefficients
    print(f'MSE2 : {round(mean_squared_error(y,y_hat),1)}') 
    print(f'RMSE2 : {round(mean_squared_error(y,y_hat,squared=False),1)}')

    # chart
    plt.title('Observations with a line')
    plt.scatter(x,y) # scatter
    plt.plot(x, y_hat, 'r--') # line

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 
# MAGIC We want to fit a model that looks like this: 
# MAGIC $$\widehat{y} = \beta_0 + \beta_1 x_1 ,$$ 
# MAGIC where \\(\beta_0\\) is a bias term and \\(\beta_1\\) is the slope of the line.
# MAGIC 
# MAGIC Use our user-defined function `plot_regression()` and try different values of bias and coeff of the regression line. Observe the change in the values of the cost function and the behavior of the line with the change of the parameters. 
# MAGIC - What does a line look like if the coeff is positive, negative or zero?
# MAGIC - What is the influence of the bias term on the line?
# MAGIC - Can you guess a suitable set of parameters? 
# MAGIC - What are the units of the MSE and RMSE in relation to dependent variable y? Which one is more intuitive to use for interpretation?

# COMMAND ----------

# Task: Try different values of bias and coeffs

# plot_regression(x, y, bias=..., coeff=...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normal Equation
# MAGIC 
# MAGIC The function `normal_eq()`  computes \\(\widehat{\beta}= (X^T X)^{-1} X^T y \\). It takes as input X and y and returns optimal values for the bias (intercept) term and the coefficient (slope).

# COMMAND ----------

def normal_eq(x,y): 
    """
    The function analytically computes the optimal set of coefficients given x and y. 
    A vector of ones is appended as the first vector of matrix x to take into account the bias term. 
    ---------------
    params:
    - x: input features matrix
    - y: target variable
    returns: 
    - beta_hat: optimal set of coefficients for linear regression
    """
    X = np.c_[np.ones(len(x)),x]
    beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print(f"Optimal set of coefficients: {beta_hat}")
    return(beta_hat)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - Use our user-defined function `normal_eq()` on the input features x and the output vector y.
# MAGIC - Use the returned values of the bias and coef in the `plot_regression` function.
# MAGIC 
# MAGIC Was your guess for the bias and coeff value from the previous exercise close enough?

# COMMAND ----------

# Task: use normal_eq() and find coeffs

#..., ... = normal_eq(x, y) 
#plot_regression(..., ..., ..., ...)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sklearn Linear Regression
# MAGIC 
# MAGIC Here we explore the linear regression from scikit learn for the first time. Help yourself with [examples from the documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) if needed: 
# MAGIC 
# MAGIC ### Exercise
# MAGIC - Use `LinearRegression()` from `sklearn.linear_model` to fit the linear regression model on x, y. Look at the examples section in the documentation if you need help.
# MAGIC - Return the coefficients for slope and intercept of the regression line. You can find them in the attributes section of the documentation. Store the values in the variables `lr_coef` and `lr_intercept`.
# MAGIC Are these values the same as the ones from the normal equation?

# COMMAND ----------

# Task: implement linear regression

lr = ...
lr.fit(..., ...)
lr_coef, lr_intercept = lr...., lr....
print(f'Slope: {lr_coef}\nBias: {lr_intercept}')

# COMMAND ----------

# MAGIC %md
# MAGIC - Next, predict the value of the new observation. If needed use the documentation for some examples.

# COMMAND ----------

# Task: predict the value of new observations

x_new = [[1.5], [0]]
lr.predict(...)

# COMMAND ----------

# MAGIC %md
# MAGIC - Lastly, return the score of the model on x and y. You can read more about the score in the documentation. The best value is 1. Usually it is between 0 and 1 but it can be also negative. The score is the R-squared metric that can be used for the evaluation of the model.

# COMMAND ----------

# Task: predict the value of new observations


# COMMAND ----------

# MAGIC %md
# MAGIC ## Outliers
# MAGIC 
# MAGIC We will now add outliers to our dataset and save them under the names x2 and y2.

# COMMAND ----------

x2 = np.append(x,[np.min(x)-0.1, np.min(x), np.min(x)-0.15]).reshape([-1,1])
y2 = np.append(y, [-400,-300,-350]).reshape([-1,1])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC Fit the linear regression to x2, y2 and store the bias in the variable `lr_outlier_intercept` and the slope in the variable `lr_outlier_coef`.

# COMMAND ----------

# Task: Fit linear regression


# COMMAND ----------

# MAGIC %md
# MAGIC You can observe on the chart how outliers influence the regression line. Outlier treatment should be one of the first steps done before fitting a linear regression model, otherwise the results can be biased.

# COMMAND ----------

plt.scatter(x2,y2)
axes = plt.gca()
x_vals2 = np.array(axes.get_xlim())
y_vals = lr_intercept + lr_coef * x_vals2
y_vals2 = lr_outlier_intercept.reshape([1,]) + lr_outlier_coef.reshape([1,]) * x_vals2
plt.plot(x_vals2, y_vals, 'r--', label='original regression line')
plt.plot(x_vals2, y_vals2, 'b--', label='regression line with outliers')
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Multiple Linear Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset
# MAGIC 
# MAGIC Load sklearn's inbuilt dataset for regression. If you want you can read the description of the dataset [here](http://lib.stat.cmu.edu/datasets/boston). It has 13 attributes that can be used for predicting house prices. 

# COMMAND ----------

raw_df = pd.read_csv('Data/Boston.csv')

y = pd.DataFrame(raw_df['target'])
x = pd.DataFrame(raw_df.iloc[:,1:-1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Test Split
# MAGIC 
# MAGIC ### Exercise 
# MAGIC Use the function `train_test_split()` to split the training data into training and testing datasets.

# COMMAND ----------

# Task: use function train_test_split() to split the training data into training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(..., ..., random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit the Model
# MAGIC ### Exercise
# MAGIC - Instantiate the linear regression model.
# MAGIC - Fit the model to the training data.
# MAGIC - Print the value of the intercept of the model.
# MAGIC - Return the values of the coefficients and save them under the variable `model_coef`.

# COMMAND ----------

# Task: instantiate the model


# Task: fit the model to x, y


print(f"intercept: {...}")


# COMMAND ----------

# MAGIC %md
# MAGIC We can interpret whether the given feature influences the prediction negatively or positively based on the sign of the coefficient. Also, if all the other variables are unchanged we can see how this single variable affects the output by changing it by 1 unit.

# COMMAND ----------

df_coefs = pd.DataFrame(model_coef, index = ["Coefficient"], columns=X_train.columns)
display(df_coefs)
print(f'''For example, if you have an observation where LSTAT is equal to 50 and another one that has a value 
of LSTAT 51 and all other variables are the same, or if the variable LSTAT is changed for the investigated observation 
with other variables unchanged the effect on the target would be {np.round(df_coefs['LSTAT'][0],5)}''')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction / Model Evaluation
# MAGIC ### Exercise
# MAGIC 
# MAGIC Predict on `X_test` and store the values in `y_hat`.

# COMMAND ----------

# Task: predict on X_test and store the values into y_hat


# COMMAND ----------

# MAGIC %md
# MAGIC Below is the plot of predictions against real values. Most of the data points lie around a diagonal line. This means that the predictions seem to be in line with the actual values, e.g. if the real value was 20, the prediction is also almost 20.

# COMMAND ----------

plt.scatter(y_hat, y_test)
plt.plot([0,50],[0,50],c='b')
plt.xlabel("predictions")
plt.ylabel("true test values")
plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC Let's have a look at some metrics. Compute and save the following metrics on the test set:
# MAGIC 
# MAGIC - MSE -  `mean_squared_error` from `sklearn.metrics` (see ?mean_squared_error)
# MAGIC - RMSE - `mean_squared_error`  from `sklearn.metrics` with the parameter `squared` set to `False`
# MAGIC - MAE (Mean absolute error) - `mean_absolute_error`  from `sklearn.metrics` (see ?mean_absolute_error)
# MAGIC - R2 (score) - this is an attribute of `LinearRegression()`

# COMMAND ----------

# Task: Compute and save the metrics on the test set

mse = mean_squared_error(y_test, y_hat)
rmse = mean_squared_error(..., ..., ...)
mae = ...(..., ...)
r2 = model.score(..., ...) # the same as r2_score(y_test, y_hat)

print(f"MSE: {np.round(mse, 1)}")
print(f"RMSE: {np.round(rmse, 1)}")
print(f"MAE: {np.round(mae, 1)}")
print(f"R2: {np.round(r2, 1)}") 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scaling
# MAGIC 
# MAGIC ### Exercise
# MAGIC If features are not scaled appropriately, the intercept tells you what the expected value for the target variable would be if all the variables were equal to 0. This might be unrealistic for many features such as weight, size of the house, distance to the sea, etc. 
# MAGIC When features are scaled correctly, the intercept can be interpreted as the expected value of a target variable when all the features are equal to their averages.
# MAGIC 
# MAGIC For the next exercise you will scale your features using `StandardScaler()`.
# MAGIC - Instantiate `StandardScaler()` from `sklearn.preprocessing`.
# MAGIC - Fit the scaler to `X_train` and transform it. Save the transformed values into `X_train_scaled`.
# MAGIC - Transform the `X_test` data with the fitted scaler and save the transformed values into `X_test_scaled`.

# COMMAND ----------

# Task: Scaling

scaler = ...
X_train_scaled = scaler.fit_transform(...)
X_test_scaled = scaler.transform(...)

# COMMAND ----------

# MAGIC %md
# MAGIC If you scaled properly you will see in the boxplot chart on the right that the distribution of the variables are concentrated around zero and that the variance is similar for all the variables.

# COMMAND ----------

plt.subplot(1,2,1)
plt.title('Boxplots for original features')
sns.boxplot(data=X_train)
plt.subplot(1,2,2)
plt.title('Boxplots for scaled features')
sns.boxplot(data=X_train_scaled);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC - Fit a linear regression model to `X_train_scaled`, `y_train`.
# MAGIC - Return the value for the intercept.
# MAGIC - Return the values for the coefficients and save them to the variable `model_coef`.

# COMMAND ----------

# Task: Fit a linear regression, return values and save them to the variable model_coef

...
...
print(f"intercept: {...}")
...

# COMMAND ----------

df_coefs = pd.DataFrame(model.coef_.T, index = X_train.columns, columns=["Coefficient"]) \
            .sort_values("Coefficient") \
            .T
display(df_coefs)

# COMMAND ----------

# MAGIC %md
# MAGIC After applying `StandardScaler()` it is easier to compare coefficients with each other as the features all have the same scale. In our example the positive effect of the 'RM' variable on the output is a little bit smaller than the negative effect of the 'LSTAT' variable. In that vein, we can order the coefficients by their absolute magnitudes to understand the influence of the variables on the result. 
# MAGIC 
# MAGIC We could then try to fit the model with the variables that have the highest coefficient values in absolute terms.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 
# MAGIC 
# MAGIC - Predict with the fitted model on `X_test_scaled`.
# MAGIC - Calculate MSE, RMSE, MASE and R2. Did it change compared to the unscaled version of linear regression?

# COMMAND ----------

# Task

...
mse = ...
rmse = ...
mae = ...
r2_score = ...

print(f"MSE: {np.round(mse,1)}")
print(f"RMSE: {np.round(rmse,1)}")
print(f"MAE: {np.round(mae,1)}")
print(f"R2: {np.round(r2_score,1)}") # the same as r2_score(y_test, y_hat)

# COMMAND ----------

# MAGIC %md
# MAGIC The scaling **does not have an effect on the performance of the model**. It only helps with interpretability of the coefficients and changes the meaning of the intercept. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Result Analysis
# MAGIC Now that we have fitted a model on the standardized features and calculated the different scores, we need to analyse these results.
# MAGIC 
# MAGIC ### Residuals
# MAGIC ### Exercise
# MAGIC - Calculate residuals by deducting `y_test` from `y_hat`.

# COMMAND ----------

# Task: Calculate residuals by deducting y_test from y_hat

residuals = ...

# COMMAND ----------

# MAGIC %md
# MAGIC One of the assumptions of a linear regression is that the **residuals are normally distributed**. 
# MAGIC 
# MAGIC From the histogram below it seems that the residuals are almost normally distributed. If this is actually the case can be tested with e.g. the Kolmogorov-Smironov test or the Shapiro-Wilk test. It is also possible to draw a quantile-quantile plot. We could investigate outliers to check if they impact the residuals. 

# COMMAND ----------

plt.subplot(1,2,1)
plt.scatter(y_test, residuals)
plt.ylabel("residuals")
plt.xlabel("y_test")
plt.title("Residuals against true values")

plt.subplot(1,2,2)
plt.hist(residuals)
plt.xlabel("residuals")
plt.ylabel("frequency")
plt.title("Residuals histogram")
plt.show();

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise – The Dependent Variable
# MAGIC 
# MAGIC The lower long tail on the histogram might be also due to the distribution of the dependent variable y. When doing exploratory data analysis we should look only at the train set so that we do not detect patterns from the test set. The test set is set aside and is used only for evaluation.
# MAGIC 
# MAGIC - Create a boxplot of `y_train`.
# MAGIC - Plot a histogram of `y_train`.

# COMMAND ----------

# Task - create a boxplot


# COMMAND ----------

# MAGIC %md
# MAGIC It seems that there are some outliers on the upper part. This is expected as there can be some very expensive houses.  

# COMMAND ----------

# Task - create a histogram


# COMMAND ----------

# MAGIC %md
# MAGIC The distribution of the target variable seems to be rather bimodal. Linear regression does not perform the best for such distributions. However, if we did not take into consideration the outliers, the distribution would be closer to a normal distribution. However, are they really outliers? 
# MAGIC So the question is if we should **transform the dependent variable**. We could also consider creating two separate linear models, one for the usual values and one for the values on the upper end. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise – Outliers
# MAGIC 
# MAGIC Might the skewness of the residuals be due to the outliers in the explanatory features?
# MAGIC 
# MAGIC - Draw a boxplot for `X_train`. You can use for example a boxplot from the seaborn library.

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC There seem to be quite some outliers, e.g. in CRIM, ZN and PTRATIO B. These outliers should be treated.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise – Multicollinearity
# MAGIC 
# MAGIC Correlated features might cause the model to be quite unstable. 
# MAGIC 
# MAGIC - Calculate the correlations between all the variables with the `corr()` method called on `X_train`. Save the output into variable `corr`.

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC Below you can see correlations of the variables between each other on the left chart. Right chart shows absolute values of correlations. 

# COMMAND ----------

# charts for correlation
plt.subplot(1,2,1)
plt.title('Correlations')
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask = mask, annot=True, cmap='Reds');

plt.title('Absolute value of correlations')
plt.subplot(1,2,2)
corr_abs = corr.abs()
sns.heatmap(corr_abs, mask = mask, annot=True, cmap='Reds');

# COMMAND ----------

# MAGIC %md
# MAGIC There is a strong correlation between some features, e.g. TAX and RAD, DIS and NOX, DIS and INUDS, DIS and AGE, LSTAT and RM. **Highly correlated features in linear models can cause instability of the model**. Thus it should be tested how the models perform if we remove such features. Usually one of the features with a correlation higher than some threshold is removed from the feature set. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linearity – Exercise
# MAGIC 
# MAGIC Since we work with a linear model, there should be a linear relationship between X and y. Let's do some simple checks. 
# MAGIC 
# MAGIC - Compute the correlation between `X_train` and `y_train`. Hint: you can concatenate `X_train` and `y_train` first on `axis=1` and calculate the correlation on the concatenated data frame.

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise
# MAGIC 
# MAGIC There is a strong negative correlation between the target variable and the 'LSTAT' feature, as well as a strong positive correlation between the target variable and the 'RM' feature. 
# MAGIC 
# MAGIC - Draw a scatter plot showing the dependency between X and y not only for the two most correlated features 'LSTAT' and 'RM' but also for the other features to investigate if there is any pattern.

# COMMAND ----------

# Task


# COMMAND ----------

# MAGIC %md
# MAGIC We already know that there is a strong correlation between the target and the variable 'LSTAT' but the relationship seems to be rather non-linear. We can transform the variable 'LSTAT' for example with a negative logarithm or x-squared function to get a linear relationship between the modified variable and y. 
# MAGIC 
# MAGIC The relationship with the second highest correlated feature 'RM' seems to be linear.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model only with the 2 strongest variables 
# MAGIC ### Exercise
# MAGIC - Fit the model only with the features 'LSTAT' and 'RM'.
# MAGIC - Predict and return the RMSE.

# COMMAND ----------

# Task

...
rmse = 
print(f"RMSE: {np.round(rmse,1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC This model has a slightly worse performance than the original model with an RMSE of around 4.7. However, interpretaion of the model with 2 variables is easier and usually the generalization of the simpler model is also better. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2nd degree polynomial features
# MAGIC 
# MAGIC We have seen in the scatter plots that there likely is a non-linear relationship between 'LSTAT' and the target variable. It makes sense to test the model with the 2 variables as above but also adding polynomial features of the 2nd degree and an interaction term between the two variables. Run the cell below to see the performance of such a model.

# COMMAND ----------

poly = PolynomialFeatures(2)
x_train_poly = poly.fit_transform(X_train[['LSTAT', 'RM']])
x_test_poly = poly.transform(X_test[['LSTAT', 'RM']])
  
model_poly = LinearRegression(fit_intercept=False)
model_poly.fit(x_train_poly, y_train)
    
y_hat = model_poly.predict(x_test_poly)
print(f"RMSE: {mean_squared_error(y_test, y_hat, squared=False)}")
# depending on the version of sklearn, this will cause an error
# in that case, replace "get_feature_names_out" with "get_feature_names"
pd.DataFrame(model_poly.coef_, index = ["Coefficient"], columns = poly.get_feature_names_out())

# COMMAND ----------

# MAGIC %md
# MAGIC This model has so far the best performance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps to consider
# MAGIC How can we improve our current model even further?
# MAGIC 
# MAGIC - Treatment of the y variable - Should we use two different models? Should we transform y?
# MAGIC - Outliers need to be treated.
# MAGIC - Feature selection methods.
# MAGIC - Removal of correlated features.
# MAGIC - Feature engineering.
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------
# MAGIC Material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)

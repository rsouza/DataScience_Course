# Databricks notebook source
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # Clustering

# COMMAND ----------

# MAGIC %md
# MAGIC ## K-means

# COMMAND ----------

# MAGIC %md
# MAGIC The scikit-learn library has an implementation of the _k-means algorithm_. Let’s apply it to a set of randomly generated blobs whose labels we throw away. 
# MAGIC 
# MAGIC The `make_blobs()` function generates a data set for clustering. You can read more about how this works in the [documention](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).
# MAGIC 
# MAGIC __TO DO__: Find out how many instances with how many features were generated.

# COMMAND ----------

# Task
from sklearn.datasets import make_blobs
X,y = make_blobs(random_state=42) 
# Your code starts here
print(X.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC Plot the points X with a scatter plot

# COMMAND ----------

plt.scatter(X[:,0],X[:,1]);

# COMMAND ----------

# MAGIC %md
# MAGIC In this toy example you can guess the number of clusters by eye. Let's see if the k-means algorithm can recover these clusters as well. 
# MAGIC 
# MAGIC __TO DO__: Import `KMeans` and create an instance of the k-means model by giving it 3 as a hyperparameter for the number of clusters. Fit the model to your dataset X.    
# MAGIC Notice that we do not feed the labels y into the model! 
# MAGIC 
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# COMMAND ----------

# Task
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
model.fit(X)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Assign the centroids to the variable `centroids` and print it. For this use the KMeans model's attribute `cluster_centers_`. 
# MAGIC 
# MAGIC - The centroids are important because they are what enables KMeans to assign new, previously unseen points to the existing clusters!

# COMMAND ----------

# Task
centroids = model.cluster_centers_
print(centroids)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Assign the predicted class labels to the variable `labels` and print it. For this use the KMeans model's attribute `labels_`.

# COMMAND ----------

# Task
labels = model.labels_
print(labels)

# COMMAND ----------

# MAGIC %md
# MAGIC Plot the points X with a scatter plot with colors equal to labels. The centroids are plotted with a red dot.

# COMMAND ----------

plt.scatter(X[:,0],X[:,1], c=labels);
plt.scatter(centroids[:,0], centroids[:,1], s=100, color="red"); # Show the centres

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Return KMeans' performance measure _inertia_  using the attribute `inertia_`.
# MAGIC 
# MAGIC - inertia = Sum of squared distances of samples to their closest cluster center. The lower the inertia the better.

# COMMAND ----------

# Task
model.inertia_

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we select the number of clusters where inertia stops decreasing significatly. (This is only a rule of thumb.)

# COMMAND ----------

inertia = []
for k in range(1, 15):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    inertia.append(km.inertia_)
    
plt.plot(range(1, 15), inertia, marker='s');
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('$J(C_k)$');

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Use the `.predict()` method of the model to predict the cluster labels of `new_points`. Assign them to a new variable named `new_labels`.    
# MAGIC Notice that KMeans can assign previously unseen points to the clusters it has already found!

# COMMAND ----------

# Task
new_points = np.array([[-4, 5], [-6, -2.5], [4, -2.5], [0, 10]])
new_labels = np.array(model.predict(new_points))
print(new_labels)

plt.scatter(X[:,0],X[:,1], c=labels)
plt.scatter(new_points[:,0], new_points[:,1], c=new_labels, marker = 'X', s = 300);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scaling numerical variables
# MAGIC 
# MAGIC Read the data set below.

# COMMAND ----------

# https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
customers = pd.read_csv('../data/Mall_Customers.csv')
customers.set_index('CustomerID', inplace = True)
customers['Annual Income (k$)'] = customers['Annual Income (k$)']*1000
customers.columns = ['gender', 'age', 'annual_income_$', 'spending_score']
display(customers.head(5))
print(customers.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC THe 'Annual Income (k$)' has a very different scale compared to the other two numerical features. 
# MAGIC 
# MAGIC For distance based methods scaling is important if the original features have very different scales. In this example we will scale all numerical features with [`StandardScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). It standardizes features by shifting the mean to zero and scaling everything to a unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = customers.select_dtypes(exclude='object').columns # get numerical columns
customers[num_cols] = scaler.fit_transform(customers[num_cols]) # apply scaler to numerical columns
customers.head()

# COMMAND ----------

customers.info()

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Fit the KMeans model to the customers dataset.    
# MAGIC **Note:** You will get an **error**. Don't worry, this one is on purpose. Proceed with the next cell after the error.

# COMMAND ----------

# Task
KMeans(n_clusters=2, random_state=42).fit(customers)

# COMMAND ----------

# MAGIC %md
# MAGIC What is the problem here? We have to transform non-numerical columns to numerical ones as it is not possible to calculate a distance between non-numerical features.

# COMMAND ----------

# use get_dummies from pandas to encode values from non-numerical columnns to numerical
# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
customers = pd.get_dummies(customers, drop_first=True)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Again, try to fit the KMeans model to the customers data. Now there should not be an error.

# COMMAND ----------

# Task
KMeans().fit(customers).labels_

# COMMAND ----------

# MAGIC %md
# MAGIC ## DBSCAN  
# MAGIC #### This part is voluntary. Do it if you have managed to finish the first part before the time limit.  
# MAGIC 
# MAGIC DBSCAN: Density-Based Spatial Clustering of Applications with Noise. This algorithm finds core samples of high density and expands clusters from them. It is good for data which contains clusters of similar density.

# COMMAND ----------

from sklearn.cluster import DBSCAN
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC Import the dataset and plot it. Clearly there should be two clusters:

# COMMAND ----------

x_moons, y_moons = datasets.make_moons(n_samples=1000, noise=0.05,random_state=42)
plt.scatter(x_moons[:, 0], x_moons[:, 1], c=y_moons);

# COMMAND ----------

# MAGIC %md
# MAGIC KMeans fails to find appropriate clusters in this example as it searches for **convex shapes**. Take a look at the plot below.

# COMMAND ----------

kmeans_labels = KMeans(2).fit(x_moons).labels_
plt.scatter(x_moons[:, 0], x_moons[:, 1], c=kmeans_labels);

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Initiate DBSCAN  with `eps=0.05` and` min_samples=5` and assign it to the variable `dbscan`. Then fit the model to `x_moons`.

# COMMAND ----------

# Task
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(x_moons)

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: Return the labels of `dbscan` with the attribute `labels_`. Assign the labels to the variable `dbscan_labels`.

# COMMAND ----------

# Task
dbscan_labels = dbscan.labels_
dbscan_labels

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that there are labels with value -1. This -1 denotes outliers.

# COMMAND ----------

unique, counts = np.unique(dbscan_labels, return_counts=True)
display(pd.DataFrame(np.asarray((unique, counts)).T, columns = ['labels', 'frequency']))

# COMMAND ----------

# MAGIC %md
# MAGIC The clusters and outliers on a plot

# COMMAND ----------

plt.scatter(x_moons[:, 0], x_moons[:, 1], c=dbscan_labels, alpha=0.7);

# COMMAND ----------

# MAGIC %md
# MAGIC __TO DO__: The indices of the core instances are available through the `core_sample_indices_` attribute. Assign them to the `comps_idx` variable.

# COMMAND ----------

# Task
comps_idx = dbscan.core_sample_indices_

# COMMAND ----------

# MAGIC %md
# MAGIC The core instances are stored in the `components_` attribute. Below we create a mask for indices which are core instances.

# COMMAND ----------

print(dbscan.components_)

comps_idx_boolean = np.array([(i in comps_idx) for i in range(len(x_moons))]) # this creates a boolean mask for core instances 

# COMMAND ----------

plt.scatter(x_moons[comps_idx_boolean, 0], x_moons[comps_idx_boolean, 1], c='r', alpha=0.3, label='core')
plt.scatter(x_moons[~comps_idx_boolean, 0], x_moons[~comps_idx_boolean, 1], c='b', alpha=0.6, linewidths = 0.1, label='outliers')
plt.legend()
plt.title('Core vs non-core instances');

# COMMAND ----------

# MAGIC %md
# MAGIC The DBSCAN clustering did not return the expected results. Let's try different `eps` and `min_samples` hyperparameters. 
# MAGIC 
# MAGIC __TO DO__: Instantiate and fit DBSCAN again with `eps=0.2` and `min_samples=5`. Plot the resulting clusters.

# COMMAND ----------

# Task
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(x_moons)
dbscan_labels = dbscan.labels_
plt.scatter(x_moons[:, 0], x_moons[:, 1], c=dbscan_labels, alpha=0.7);

# COMMAND ----------

# MAGIC %md
# MAGIC **Notice that DBSCAN class does not have a `.predict()` method**, although it has a `fit_predict()` method. This is because DBSCAN cannot predict a cluster for a new instance!

# COMMAND ----------

dbscan.predict(x_moons)

# COMMAND ----------

# MAGIC %md
# MAGIC Some material adapted for RBI internal purposes with full permissions from original authors. [Source](https://github.com/zatkopatrik/authentic-data-science)

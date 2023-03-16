# Databricks notebook source
# MAGIC %md
# MAGIC ## Introduction to Natural Language Processing tasks  

# COMMAND ----------

!pip install -U eli5

# COMMAND ----------

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import eli5

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Classification
# MAGIC #### "Traditional" Text Classification with Scikit-learn
# MAGIC In this notebook, we're going to experiment with a few "traditional" approaches to text classification. These approaches pre-date the deep learning revolution in Natural Language Processing, but are often quick and effective ways of training a text classifier.
# MAGIC 
# MAGIC #### Data
# MAGIC We are going to work with the **20 Newsgroups data set**, a classic collection of text documents that is often used as a benchmark for text classification models. The set contains texts about various topics, ranging from computer hardward to religion. Some of the topics are closely related to each other (such as "IBM PC hardware" and "Mac hardware"), while others are very different (such as "religion" or "hockey"). The 20 Newsgroups comes shipped with the Scikit-learn machine learning library, our main tool for this exercise. It has been split into training set of 11,314 texts and a test set of 7,532 texts.

# COMMAND ----------

train_data = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test')

print("Training texts:", len(train_data.data))
print("Test texts:", len(test_data.data))

# COMMAND ----------

train_data.target_names

# COMMAND ----------

text_num = 10
print(f"The topic is {train_data.target_names[train_data.target[text_num]]}")
print()
print(train_data.data[text_num])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preprocessing  
# MAGIC 
# MAGIC The first step in the development of any NLP model is text preprocessing. This means we're going to transform our texts from word sequences to feature vectors. These feature vectors each contain the values of' a large number of features.  
# MAGIC 
# MAGIC In this experiment, we're going to work with so-called **"bag-of-word"** approaches. Bag-of-word methods treat every text as an unordered collection of words (or optionally, _ngrams_), and the raw feature vectors simply tell us how often each word (or ngram) occurs in a text. In Scikit-learn, we can construct these raw feature vectors with `CountVectorizer`, which tokenizes a text and counts the number of times any given text contains every token in the corpus.  
# MAGIC 
# MAGIC However, these raw counts are not very informative yet. This is because the raw feature vectors of most texts in the same language will be very similar. For example, most texts in English contain many instances of relatively uninformative words, such as a, the or be. Instead, what we're interested in are words like computer or hardware: words that occur often in one text, but not very often in the corpus as a whole. Therefore we're going to weight all features by their **tf-idf score**, which counts the number of times every token appears in a text and divides it by (the logarithm of) the percentage of corpus documents that contain that token. This weighting is performed by Scikit-learn's `TfidfTransformer`.  
# MAGIC 
# MAGIC To obtain the weighted feature vectors, we combine the `CountVectorizer` and `TfidfTransformer` in a Pipeline, and fit this pipeline on the training data. We then transform both the training texts and the test texts to a collection of such weighted feature vectors. Scikit-learn also has a `TfidfVectorizer`, which achieves the same result as our pipeline.

# COMMAND ----------

def clean_and_tokenize_text(news_data):
    """Cleans some issues with the text data
    Args:
        news_data: list of text strings
    Returns:
        For each text string, an array of tokenized words are returned in a list
    """
    cleaned_text = []
    for text in news_data:
        x = re.sub('[^\w]|_', ' ', text)  # only keep numbers and letters and spaces
        x = x.lower()
        x = re.sub(r'[^\x00-\x7f]',r'', x)  # remove non ascii texts
        tokens = [y for y in x.split(' ') if y] # remove empty words
        tokens = ['[number]' if x.isdigit() else x for x in tokens]
        tokens =  ' '.join(tokens)
        # As an exercise, try stemming each token using python's nltk package.
        cleaned_text.append(tokens)
    return cleaned_text

train = clean_and_tokenize_text(train_data.data)
test = clean_and_tokenize_text(test_data.data)

# COMMAND ----------

preprocessing = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])
  
print("Preprocessing training data...")
#train_preprocessed = preprocessing.fit_transform(train_data.data)
train_preprocessed = preprocessing.fit_transform(train)

print("Preprocessing test data...")
#test_preprocessed = preprocessing.transform(test_data.data)
test_preprocessed = preprocessing.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Training
# MAGIC 
# MAGIC Next, we train a text classifier on the preprocessed training data. We're going to experiment with three classic text classification models: Naive Bayes, Support Vector Machines and Logistic Regression. 
# MAGIC 
# MAGIC [Naive Bayes classifiers](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) are extremely simple classifiers that assume all features are independent of each other. They just learn how frequent all classes are and how frequently each feature occurs in a class. To classify a new text, they simply multiply the probabilities for every feature \\(x_i\\) given each class \\(C\\) and pick the class that gives the highest probability: 
# MAGIC 
# MAGIC $$ \hat y = argmax_k \, [ \, p(C_k) \prod_{i=1}^n p(x_i \mid C_k)\, ]  $$
# MAGIC 
# MAGIC Naive Bayes Classifiers are very quick to train, but usually fall behind in terms of performance.
# MAGIC 
# MAGIC [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) are much more advanced than Naive Bayes classifiers. They try to find the hyperplane in the feature space that best separates the data from the different classes. They do so by picking the hyperplane that maximizes the distance to the nearest data point on each side. When the classes are not linearly separable, SVMs map the data into a higher-dimensional space where a linear separation can hopefully be found. SVMs often achieve very good performance in text classification tasks.
# MAGIC 
# MAGIC [Logistic Regression models](https://en.wikipedia.org/wiki/Logistic_regression), finally, model the log-odds \\(l\\), or \\(\log[p\,/\,(1-p)]\\), of a class as a linear model and estimate the parameters \\(\beta\\) of the model during training: 
# MAGIC 
# MAGIC \\(l = \beta_0 + \sum_{i=1}^n \beta_i x_i\\)
# MAGIC 
# MAGIC Like SVMs, they often achieve great performance in text classification.
# MAGIC 
# MAGIC ##### Simple training
# MAGIC We train our three classifiers in Scikit-learn with the fit method, giving it the preprocessed training text and the correct classes for each text as parameters.

# COMMAND ----------

nb_classifier = MultinomialNB()
svm_classifier = LinearSVC()
lr_classifier = LogisticRegression(multi_class="ovr")

print("Training Naive Bayes classifier...")
nb_classifier.fit(train_preprocessed, train_data.target)

print("Training SVM classifier...")
svm_classifier.fit(train_preprocessed, train_data.target)

print("Training Logistic Regression classifier...")
lr_classifier.fit(train_preprocessed, train_data.target)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's find out how well each classifier performs. To find out we have each classifier `predict` the label for all texts in our preprocessed test set.

# COMMAND ----------

nb_predictions = nb_classifier.predict(test_preprocessed)
lr_predictions = lr_classifier.predict(test_preprocessed)
svm_predictions = svm_classifier.predict(test_preprocessed)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can compute the accuracy of each model: the proportion of test texts for which the predicted label is the same as the target label. The Naive Bayes classifier assigned the correct label in 77.4% of the cases, the logistic regression model achieves an accuracy of 82.8%, and the Support Vector Machine got the label right 85.3% of the time.

# COMMAND ----------

print("NB Accuracy:", np.mean(nb_predictions == test_data.target))
print("LR Accuracy:", np.mean(lr_predictions == test_data.target))
print("SVM Accuracy:", np.mean(svm_predictions == test_data.target))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grid search
# MAGIC 
# MAGIC Still, it's a bit too early to announce the winner. It's very likely we haven't yet got the most from our classifiers. When we trained them above, we just used the default values for most hyperparameters. However, these hyperparameter values can have a big impact on accuracy. Therefore we want to explore the parameter space a bit more and find out what hyperparameter values give the best results. We do this with a so-called grid search. In a grid search we define a grid of hyperparameter values that we want to explore. Scikit-learn then steps to this grid to find the best combination. It does this with **n-fold cross-validation**: for each parameter combination in the grid, it fits a predefined number of models (**n**, the `cv` parameter in `GridSearchCV`. It splits up the training data in **n folds**, fits a model on all but one of these folds, and tests it on the held-out fold. When it has done this **n** times, it computes the average performance, and moves on. It performs the full hyperparameter grid in this way and keeps the model with the best average performance over the folds.
# MAGIC 
# MAGIC In this example, we'll experiment with the **C hyperparameter**. C controls the degree of regularization in support vector machines and logistic regression. Regularization combats overfitting by imposing a penalty on large parameter values in the model. The lower the C value, the more regularization is applied.

# COMMAND ----------

parameters = {'C': np.logspace(0, 3, 10)}
parameters = {'C': [0.1, 1, 10, 100, 1000]}

print("Grid search for SVM")
svm_best = GridSearchCV(svm_classifier, parameters, cv=3, verbose=1, n_jobs=-1)
svm_best.fit(train_preprocessed, train_data.target)

print("Grid search for logistic regression")
lr_best = GridSearchCV(lr_classifier, parameters, cv=3, verbose=1, n_jobs=-1)
lr_best.fit(train_preprocessed, train_data.target)

# COMMAND ----------

# MAGIC %md
# MAGIC When the grid search has been completed, we can find out what hyperparameter values led to the best-performing model.

# COMMAND ----------

print("Best LR parameters:")
print(lr_best.best_params_)

print("Best SVM Parameters")
print(svm_best.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see if these best models now perform any better on our test data. For the SVM, the default setting seems to have worked best: our other values didn't lead to a higher accuracy. For logistic regression, however, the default  𝐶  value was clearly not the most optimal one. When we increase  𝐶  to  1000 , the logistic regression model performs almost as well as the SVM.

# COMMAND ----------

best_lr_predictions = lr_best.predict(test_preprocessed)
best_svm_predictions = svm_best.predict(test_preprocessed)

print("Best LR Accuracy:", np.mean(best_lr_predictions == test_data.target))
print("Best SVM Accuracy:", np.mean(best_svm_predictions == test_data.target))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extensive evaluation
# MAGIC 
# MAGIC #### Detailed scores
# MAGIC 
# MAGIC So far we've only looked at the accuracy of our models: the proportion of test examples for which their prediction is correct. This is fine as a first evaluation, but it doesn't give us much insight in what mistakes the models make and why. We'll therefore perform a much more extensive evaluation in three steps. Let's start by computing the precision, recall and F-score of the best SVM for the individual classes:
# MAGIC 
# MAGIC - Precision \\(P\\) is the number of times the classifier predicted a class correctly, divided by the total number of times it predicted this class. 
# MAGIC - Recall is the proportion of documents with a given class that were labelled correctly by the classifier. 
# MAGIC - The F1-score is the harmonic mean between precision and recall: \\( 2PR/(P+R) \\)
# MAGIC 
# MAGIC The classification report below shows, for example, that the sports classes were quite easy to predict, while the computer and some of the politics classes proved much more difficult.

# COMMAND ----------

print(classification_report(test_data.target, best_svm_predictions, target_names=test_data.target_names))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Confusion matrix  
# MAGIC 
# MAGIC Second, we're going to visualize our results in even more detail, using a so-called confusion matrix. A confusion matrix helps us better understand the errors our classifier makes. Its rows display the actual labels, its columns show the predictions of our classifier. This means all correct predictions will lie on the diagonal, where the actual label and the predicted label are the same. The predictions elsewhere in the matrix help us understand what classes are often mixed up by our classifier. Our confusion matrix shows, for example, that 91 documents with the label talk.politics.misc incorrectly received the label talk.politics.guns. Similarly, our classifier sometimes fails to tell apart the two religion classes, and gets quite mixed up in the computer topics in the top left corner.

# COMMAND ----------

conf_matrix = confusion_matrix(test_data.target, best_svm_predictions)
conf_matrix_df = pd.DataFrame(conf_matrix, index=test_data.target_names, columns=test_data.target_names)

plt.figure(figsize=(15, 10))
sn.heatmap(conf_matrix_df, annot=True, vmin=0, vmax=conf_matrix.max(), fmt='d', cmap="YlGnBu")
plt.yticks(rotation=0)
plt.xticks(rotation=90)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explainability
# MAGIC 
# MAGIC Finally, we'd like to perform a more qualitative evaluation of our model by taking a look at the features to which it assigns the highest weight for each of the classes. This will help us understand if the model indeed captures the phenomena we'd like it to capture. A great Python library to do this is `eli5`, which works together seamlessly with `scikit-learn`. Its function `explain_weights` takes a trained model, a list of feature names and target names, and prints out the features that have the highest positive values for each of the targets. The results convince us that our SVM indeed models the correct information: it sees a strong link between the "atheism" class and words such as _atheism_ and _atheists_, between "computer graphics" and words such as _3d_ and _image_, and so on.

# COMMAND ----------

eli5.explain_weights(svm_best.best_estimator_, 
                     feature_names = preprocessing.named_steps["vect"].get_feature_names(),
                     target_names = train_data.target_names
                    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Conclusion  
# MAGIC 
# MAGIC This notebook has demonstrated how you can quickly train a text classifier. Although the types of models we've looked at predate the deep learning revolution in NLP, they're often a quick and effective way of training a first classifier for your text classification problem. Not only can they provide a good baseline and help you understand your data and problem better. In some cases, you may find they are quite hard to beat even with state-of-the-art deep learning models.

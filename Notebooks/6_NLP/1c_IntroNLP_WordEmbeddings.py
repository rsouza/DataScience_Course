# Databricks notebook source
# MAGIC %md
# MAGIC ## Introduction to Natural Language Processing tasks  
# MAGIC #### An Introduction to Word Embeddings

# COMMAND ----------

!pip install -U -q spacy gensim

# COMMAND ----------

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
%matplotlib inline

import os
import csv
import spacy
import gensim
import time

%matplotlib inline

from sklearn.manifold import TSNE

# COMMAND ----------

begin = time.time()

# COMMAND ----------

# MAGIC %md
# MAGIC Many people would say the breakthrough of deep learning in natural language processing started with the introduction of word embeddings. Rather than using the words themselves as features, neural network methods typically take as input dense, relatively low-dimensional vectors that model the meaning and usage of a word. Word embeddings were first popularized through the [Word2Vec](https://arxiv.org/abs/1301.3781) model, developed by Thomas Mikolov and colleagues at Google. Since then, scores of alternative approaches have been developed, such as [GloVe](https://nlp.stanford.edu/projects/glove/) and [FastText](https://fasttext.cc/) embeddings. In this notebook, we'll explore word embeddings with the original Word2Vec approach, as implemented in the [Gensim](https://radimrehurek.com/gensim/) library.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training word embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC Training word embeddings with Gensim couldn't be easier. The only thing we need is a corpus of sentences in the language of interest. For our experiments we're going to use the abstracts of all ArXiv papers in the category cs.CL (computation and language) that were published before mid-April 2021 — a total of around 25,000 documents. We tokenize these abstracts with _spaCy_.

# COMMAND ----------

class Corpus(object):

    def __init__(self, filename):
        self.filename = filename
        self.nlp = spacy.blank("en")
        
    def __iter__(self):
        with open(self.filename, "r") as i:
            reader = csv.reader(i, delimiter=",")
            for _, abstract in reader:
                tokens = [t.text.lower() for t in self.nlp(abstract)]
                yield tokens
                            
                    
documents = Corpus(os.path.join(os.getcwd(), "data/arxiv/arxiv.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC When we train our word embeddings, Gensim allows us to set a number of parameters. The most important of these are `min_count`, `window`, `vector_size` and `sg`:
# MAGIC 
# MAGIC - `min_count` is the minimum frequency of the words in our corpus. For infrequent words we just don't have enough information to train reliable word embeddings. It therefore makes sense to set this minimum frequency to at least 10. In these experiments, we'll set it to 100 to limit the size of our model even more.
# MAGIC - `window` is the number of words to the left and to the right that make up the context that word2vec will take into account.
# MAGIC - `vector_size` is the dimensionality of the word vectors. This is generally between 100 and 1000. This dimensionality often forces us to make a trade-off: embeddings with a higher dimensionality are able to model more information, but also need more data to train.
# MAGIC - `sg`: there are two algorithms to train `Word2Vec`: skip-gram and CBOW. Skip-gram tries to predict the context on the basis of the target word; CBOW tries to find the target on the basis of the context. By default, Gensim uses CBOW (`sg=0`).
# MAGIC 
# MAGIC We'll investigate the impact of some of these parameters later.

# COMMAND ----------

# https://radimrehurek.com/gensim/models/word2vec.html

model = gensim.models.Word2Vec(documents, min_count=100, window=5, vector_size=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using word embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the trained model. The word embeddings are on its `wv` attribute and we can access them by  using the token as key. For example, here is the embedding for *nlp*, with the requested 100 dimensions.

# COMMAND ----------

model.wv["nlp"]

# COMMAND ----------

# MAGIC %md
# MAGIC We can also easily find the similarity between two words. Similarity is measured as the cosine between the two word embeddings, and therefore ranges between -1 and +1. The higher the cosine, the more similar two words are. As expected, the figures below show that *nmt* (neural machine translation) is closer to *smt* (statistical machine translation) than to *ner* (named entity recognition).

# COMMAND ----------

print(model.wv.similarity("nmt", "smt"))
print(model.wv.similarity("nmt", "ner"))

# COMMAND ----------

# MAGIC %md
# MAGIC In a similar vein, we can find the words that are most similar to a target word. The words with the most similar embedding to *bert* are all semantically related to it: other types of pretrained models such as *roberta*, *mbert*, *xlm*, as well as the more general model type BERT represents (*transformer* and *transformers*), and more generally related words (*pretrained*).

# COMMAND ----------

model.wv.similar_by_word("bert", topn=10)

# COMMAND ----------

# MAGIC %md
# MAGIC Interestingly, we can look for words that are similar to a set of words and dissimilar to another set of words at the same time. This allows us to look for analogies of the type *"BERT is to a transformer like an LSTM is to ..."*. Our embedding model correctly predicts that LSTMs are a type of RNN, just like BERT is a particular type of transformer.

# COMMAND ----------

model.wv.most_similar(positive=["transformer", "lstm"], negative=["bert"], topn=1)

# COMMAND ----------

# MAGIC %md
# MAGIC Similarly, we can also zoom in on one of the meanings of ambiguous words. For example, in NLP *tree* has a very specific meaning, which is obvious from its nearest neighbours *constituency*, *parse*, *dependency* and *syntax*.

# COMMAND ----------

model.wv.most_similar(positive=["tree"], topn=10)

# COMMAND ----------

# MAGIC %md
# MAGIC However, if we specify we're looking for words that are similar to *tree*, but dissimilar to *syntax*, suddenly its standard meaning takes over, and *forest* crops up in its nearest neighbours.

# COMMAND ----------

model.wv.most_similar(positive=["tree"], negative=["syntax"], topn=10)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we can present the `WordVec` model with a list of words and ask it to identify the odd one out. It then uses the word embeddings to identify the word that is least similar to the other ones. For example, in the list *lstm cnn gru svm transformer*, it correctly identifies *svm* as the only non-neural model. In the list *bert word2vec gpt-2 roberta xlnet*, it correctly singles out *word2vec* as the only non-transormer model. In *word2vec bert glove fasttext elmo*, *bert* is singled out as the only transformer.

# COMMAND ----------

print(model.wv.doesnt_match("lstm cnn gru svm transformer".split()))
print(model.wv.doesnt_match("bert word2vec gpt-2 roberta xlnet".split()))
print(model.wv.doesnt_match("word2vec bert glove fasttext elmo".split()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plotting embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now visualize some of our embeddings. To plot embeddings with a dimensionality of 100 or more, we first need to map them to a dimensionality of 2. We do this with the popular [t-SNE](https://lvdmaaten.github.io/tsne/) method. T-SNE, short for **t-distributed Stochastic Neighbor Embedding**, helps us visualize high-dimensional data by mapping similar data to nearby points and dissimilar data to distant points in the low-dimensional space.
# MAGIC 
# MAGIC T-SNE is present in [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). To run it, we just have to specify the number of dimensions we'd like to map the data to (`n_components`), and the similarity metric that t-SNE should use to compute the similarity between two data points (`metric`). We're going to map to 2 dimensions and use the cosine as our similarity metric. Additionally, we use PCA as an initialization method to remove some noise and speed up computation. The [Scikit-learn user guide](https://scikit-learn.org/stable/modules/manifold.html#t-sne) contains some additional tips for optimizing performance. 
# MAGIC 
# MAGIC Plotting all the embeddings in our vector space would result in a very crowded figure where the labels are hardly legible. Therefore we'll focus on a subset of embeddings by selecting the 200 most similar words to a target word.

# COMMAND ----------

target_word = "bert"
selected_words = [w[0] for w in model.wv.most_similar(positive=[target_word], topn=200)] + [target_word]
embeddings = [model.wv[w] for w in selected_words] + model.wv["bert"]

mapped_embeddings = TSNE(n_components=2, metric='cosine', init='pca', square_distances=True).fit_transform(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC If we take *bert* as our target word, the figure shows some interesting patterns. In the immediate vicinity of *bert*, we find the similar transformer models that we already identified as its nearest neighbours earlier: *xlm*, *mbert*, *gpt-2*, and so on. Other parts of the picture have equally informative clusters of NLP tasks and benchmarks (*squad* and *glue*), languages (*german* and *english*), neural-network architectures (*lstm*, *gru*, etc.), embedding types (*word2vec*, *glove*, *fasttext*, *elmo*), etc.

# COMMAND ----------

plt.figure(figsize=(20,20))
x = mapped_embeddings[:,0]
y = mapped_embeddings[:,1]
plt.scatter(x, y)

for i, txt in enumerate(selected_words):
    plt.annotate(txt, (x[i], y[i]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC Word embeddings are one of the most exciting trends on Natural Language Processing since the 2000s. They allow us to model the meaning and usage of a word, and discover words that behave similarly. This is crucial for the generalization capacity of many machine learning models. Moving from raw strings to embeddings allows them to generalize across words that have a similar meaning, and to discover patterns that had previously escaped them.

# COMMAND ----------

print(f"Execution took: {((time.time() - begin)/60)} minutes")

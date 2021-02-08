# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:59:07 2020

@author: zlibn
"""
import numpy as np
import pandas as pd
# Loading gensim and nltk libraries
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

np.random.seed(2018)
import nltk
nltk.download('wordnet')

# To generate the dictionary and bow_corpus
def doc_generator(user_tensor):
    list_docs = []
    for i in range(user_tensor.shape[0]):
        user = user_tensor[i,:,:,:]
        O, D, T = np.nonzero(user)
        count = user[np.nonzero(user)]
        doc = []
        for j in range(len(O)):
            word = 'O' + str(O[j]) + '_D' + str(D[j]) + '_T' + str(T[j])
            doc += int(count[j]) * [word]
        list_docs.append(doc)
    list_docs = pd.Series(list_docs)
    
    return list_docs

# To generator the ODT Integer 
# Preprocess the headline text, saving the results as list_docs
user_tensor = user100_tensor
list_docs = doc_generator(user_tensor)
# Some properties about this list of documents
## total number of words in this list of documents
sum(len(list_docs[i]) for i in range(user_tensor.shape[0]))
## word counter for each document in this list
word_counter = [len(list_docs[i]) for i in range(user_tensor.shape[0])]

# Bag of Words on the Data set
# Create a dictionary from ‘processed_docs’ containing the number of times a word appears in the training set.

dictionary = gensim.corpora.Dictionary(list_docs)

# Filter out tokens that appear in
# less than 3 documents (absolute number) or
# more than 0.5 documents (fraction of total corpus size, not absolute number).
# after the above two steps, keep only the first 100000 most frequent tokens.

dictionary.filter_extremes(no_below=3, no_above=0.5, keep_n=100000)

# Some properties about this dictionary
## show the first 10 words
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
## count the word frequencies
word_freq = dictionary.cfs
len(word_freq)
sum(word_freq.values())


# Gensim doc2bow
# For each document we create a dictionary reporting how many
# words and how many times those words appear. Save this to ‘bow_corpus’, then check our selected document earlier.

bow_corpus = [dictionary.doc2bow(doc) for doc in list_docs]
bow_corpus[10]

# Preview Bag Of Words for our sample preprocessed document.
bow_doc_10 = bow_corpus[10]
for i in range(len(bow_doc_10)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_10[i][0], dictionary[bow_doc_10[i][0]], bow_doc_10[i][1]))

# TF-IDF
# Create tf-idf model object using models.TfidfModel on ‘bow_corpus’ and 
# save it to ‘tfidf’, then apply transformation to the entire corpus and call 
# it ‘corpus_tfidf’. Finally we preview TF-IDF scores for our first document.
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

# Running LDA using Bag of Words
# Train our lda model using gensim.models.LdaMulticore and save it to 'lda_model'
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)
# For each topic, we will explore the words occuring in that topic and its relative weight.
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# Running LDA using TF-IDF
# Train our lda model using gensim.models.LdaMulticore and save it to 'lda_model_tfidf'
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)

# Perplexity
lda_model_tfidf.log_perplexity(bow_corpus)

# For each topic, we will explore the words occuring in that topic and its relative weight.
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

# Performance evaluation by classifying sample document using LDA Bag of Words model
user = 20
print(list_docs[user])

for index, score in sorted(lda_model[bow_corpus[user]], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    
# Performance evaluation by classifying sample document using LDA TF-IDF model.
for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))

# Testing model on unseen document
unseen_trip = ['O18_D7_T0','O7_D18_T1','O18_D60_T2','O60_D6_T2']
bow_vector = dictionary.doc2bow(unseen_trip)
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

# To get the document*topic matrix
n_topics = 5
doc_topic_matrix = np.zeros((len(list_docs), n_topics))
for i in range(len(list_docs)):
    for j in range(n_topics):
        doc_topic_matrix[i,j] = lda_model[bow_corpus[i]][j][1]










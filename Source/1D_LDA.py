# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:11:16 2020

@author: zlibn
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:59:30 2020

@author: zlibn
"""
import sys
sys.path.append("D:/Google Drive/HKUST-Office/Research/4th Work/source")

import numpy as np
import pandas as pd
import random
import math
import gensim
from scipy.special import psi, loggamma, polygamma
from Telegram_chatbot import MTRobot
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def tensor2doc(user_tensor):
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


def doc_generator(list_docs):
    wordcount = []
    for i in range(len(list_docs)):
        count = len(list_docs[i])
        wordcount.append(count)
    docs = pd.concat([pd.Series(list_docs), pd.Series(wordcount)], axis=1)
    docs.columns = ['ODT','wordcount']
    dictionary = gensim.corpora.Dictionary(docs['ODT'])
    idx_corpus = [dictionary.doc2idx(doc) for doc in docs['ODT']]
    
    return docs, dictionary, idx_corpus

def maxItemNum():
    num = 0
    for u in range(M):
        if  docs.iloc[u]['wordcount'] > num: #len(docs[d].itemIdList): number of unique words in a document
            num = int(docs.iloc[u]['wordcount'])
    return num


def initial_count(num_topic, num_word):
    count_zw = np.zeros((num_topic, num_word))     # sufficient statistic for beta
    count_z = np.zeros(num_topic)
    for z in range(num_topic):
        for w in range(num_word):
            count_zw[z, w] += 1.0/num_word + random.random() 
            count_z[z] += count_zw[z, w]
    return count_zw, count_z

def initialLdaModel():
    count_zw, count_z = initial_count(K, N)
    
    beta = update_beta(count_zw, count_z)
    return count_zw, count_z, beta


# update model parameters : beta (the topic-word parameter， (real-value) log-value is actually calculated here)
# (the update of alpha is ommited)
def update_beta(count_zw, count_z):
    num_topic = count_zw.shape[0]
    num_word = count_zw.shape[1]
    beta = np.zeros((num_topic, num_word))

    for z in range(num_topic):
        for w in range(0, num_word):
            if(count_zw[z, w] > 0):
                beta[z, w] = math.log(count_zw[z, w]) - math.log(count_z[z]) # beta[z, w] = count_zw[z, w] / count_z[z] 
            else:
                beta[z, w] = -100 # beta[z, w] = 0
    return beta

def variationalInference(docs, u, gamma, phi):
    phisum = 0
    oldphi = np.zeros(K)
    digamma_gamma = np.zeros(K)
     
    # Initialization for phiO, phiD, phiT:
    for k in range(K):
        gamma[u][k] = alpha + docs.iloc[u]['wordcount'] * 1.0 / K
        digamma_gamma[k] = psi(gamma[u][k])
        for w in range(len(idx_corpus[u])): # number of unigue word
            phi[w, k] = 1.0 / K    

    for iteration in range(iterInference):
        #MTRobot.sendtext("---Variational Inference Iter {}".format(iteration))

    # To update phiD:
        #MTRobot.sendtext("Update phiD: iter {}".format(iteration))
        for w in range(len(idx_corpus[u])):
            phisum = 0
            for k in range(K):
                oldphi[k] = phi[w, k]
                phi[w, k] = digamma_gamma[k] + beta[k, idx_corpus[u][w]]
                if k > 0:
                    phisum = math.log(math.exp(phisum) + math.exp(phi[w, k]))
                else:
                    phisum = phi[w, k]
            for k in range(K):
                phi[w, k] = math.exp(phi[w, k] - phisum) # normalization
                # To update gamma
                gamma[u][k] =  gamma[u][k] + (phi[w, k] - oldphi[k]) # (1) why phi - oldphi (solved); (2) why docs[d].itemCountList[w]: since here in the code, w is the unique word from a doc, while in paper w is the single word in the doc
                digamma_gamma[k] = psi(gamma[u][k])

def perlexity(test_docs, idx_corpus, alpha, count_uz, beta):
    beta = np.exp(beta)
    num_topic = beta.shape[0]
    log_per = 0
    wordcount_sum = 0
    Kalpha = num_topic * alpha
    for u in range(len(test_docs)):
        theta = count_uz[u] / (len(test_docs[u]) + Kalpha)
        for w in range(len(test_docs[u])):
            log_per -= np.log(np.inner(beta[:,idx_corpus[u][w]], theta)) # phi[:,w]: R^(K*1)
        wordcount_sum += len(test_docs[u])
    return np.exp(log_per / wordcount_sum)

def word_doc_freq(word_id, idx_corpus):
    freq = 0
    for u in range(len(idx_corpus)):
        if word_id in idx_corpus[u]:
            freq += idx_corpus[u].count(word_id)
    return freq

def words_doc_cofreq(word1_id, word2_id, idx_corpus):
    freq = 0
    for u in range(len(idx_corpus)):
        if word1_id in idx_corpus[u] and word2_id in idx_corpus[u]:
            freq += ( idx_corpus[u].count(word1_id) * idx_corpus[u].count(word2_id) )
    return freq

def topic_k_coherence(sub_topic, idx_corpus, epsilon=1):
    """
    sub_topic: one imcomplete row in beta (the top r largest element in topic k), e.g.: sorted(range(len(betaD[k])), key=lambda x: betaD[k][x])[-r:]
    """
    tc_sum = 0
    for index, w1 in enumerate(sub_topic[1:]):
        m_index = index + 1
        sublist = sub_topic[:m_index]
        
        for w2 in sublist:
            fre_w2 = word_doc_freq(w2, idx_corpus)
            cofre_w1_w2 = words_doc_cofreq(w1, w2, idx_corpus)
            #print(f'cofre_w1{w1}_w2{w2} / fre_w2{w2}: {cofre_w1_w2}/{fre_w2}')
            if fre_w2 > 0:
                tc_sum += np.log(float(cofre_w1_w2 + epsilon) / fre_w2)
    return tc_sum

def topic_coherence(beta, toprank, idx_corpus):
    num_topic = beta.shape[0]
    TC = []
    for z in range(num_topic):
        sub_topic = sorted(range(len(beta[z])), key=lambda x: beta[z][x])[-toprank:]
        tc_sum = topic_k_coherence(sub_topic, idx_corpus, epsilon=1)
        TC.append(tc_sum)
    return TC

# In[]

user_tensor = user100_tensor
list_docs = tensor2doc(user_tensor)
docs, dictionary, idx_corpus = doc_generator(list_docs)

# In[]
num_user = docs.shape[0]
num_word = len(dictionary)
# In[1.2] Parameter Introduction and Initialization
#data = pd.read_csv('C:/Users/zlibn/Desktop/sample/original.csv')
#u_list = data['id_re']
#o_list = data['xy_rankx']
#d_list = data['xy_ranky']
#t_list = data['hour_x']
#
#num_data = data.shape[0]
#num_user = len(u_list.unique())
#num_station = max(len(o_list.unique()), len(d_list.unique()))
#num_time = len(u_list.unique())

# initialization for four multinomial parameter
# the followings are topic-word counts: they are sufficient statistic to calculate beta_O, D, T
#count_ztodu = np.zeros((J*K*L, num_user))  # for Theta
#count_zwo = np.zeros((J, num_station))     # for beta^O
#count_zwd = np.zeros((K, num_station))     # for beta^D
#count_zwt = np.zeros((L, num_time))        # for beta^T

#pz = np.zeros((J*K*L, num_user))    # Theta
#po = np.zeros((J, num_station))     # beta^O
#pd = np.zeros((K, num_station))     # beta^D
#pt = np.zeros((L, num_time))        # beta^T


# number of passengers for training
M = num_user
# number of distinct terms
# N = len(word2id)
N = num_word
# number of topic
K = 10
# iteration times of variational inference, judgment of the convergence by calculating likelihood is ommited
iterInference = 5
# iteration times of variational EM algorithm, judgment of the convergence by calculating likelihood is ommited
iterEM = 5

# initial value of hyperparameter alpha
alpha = 5
# sufficient statistic of alpha
alphaSS = 0
# the topic-word distribution (beta in D. Blei's paper)
beta = np.zeros([K, N]) #+ 1e-5
# topic-word count, this is a sufficient statistic to calculate beta
count_zw = np.zeros((K, N))     # sufficient statistic for beta^D
# topic count, sum of nzw with w ranging from [0, M-1], for calculating varphi
count_z = np.zeros(K)

# inference parameter gamma
gamma = np.zeros((M, K))
# inference parameter phi
phi = np.zeros([maxItemNum(), K])


# initialization of the model parameter varphi, the update of alpha is ommited
count_zw, count_z, beta = initialLdaModel()
# In[2] variational EM Algorithm
###############################################################################
for iteration in range(iterEM):
    MTRobot.sendtext(" - Start EM interation: {}".format(iteration))
    count_zw = np.zeros((K, N))     # sufficient statistic for beta^D
    count_z = np.zeros(K)
    count_uz = np.zeros((M, K))
    alphaSS = 0
    
    # E-Step
    #print("-start variational Inference E-step")
    MTRobot.sendtext(" - start variational Inference E-step")
    for u in range(M):
        #MTRobot.sendtext("--Passenger{}".format(u))
        variationalInference(docs, u, gamma, phi)

        gammaSum = 0
        for k in range(K):
            gammaSum += gamma[u, k]
            alphaSS += psi(gamma[u, k])
        alphaSS -= K * psi(gammaSum)
                
        # To update count_zwd, count_zd
        for w in range(len(idx_corpus[u])):
            for k in range(K):
                count_zw[k, idx_corpus[u][w]] += phi[w, k] 
                count_z[k] += phi[w, k]
                count_uz[u, k] += phi[w, k]
        
    # M-Step
    #print("start variational Inference M-step")
    MTRobot.sendtext(" - start variational Inference M-step")
    #betaO_no_g = update_beta(count_zwo, count_zo) # betaO, gradient, hessian = update_beta_w_graph(lam, count_zwo, count_zo, mu, G_net, G_poi) # update_beta(count_zwo, count_zo) 
    #betaO, gradient, hessian = update_beta_w_graph(betaO, lam, betaO_no_g, mu, G_net, G_poi)
    beta = update_beta(count_zw, count_z)
    
# In[] Model Evaluation
perlexity(docs["ODT"], idx_corpus, alpha, count_uz, beta)

topic_coherence(beta, 20, idx_corpus)




























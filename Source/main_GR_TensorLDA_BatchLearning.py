# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:57:17 2020

@author: zlibn
"""
import sys
sys.path.append("D:/Google Drive/HKUST-Office/Research/4th Work/Source")

import numpy as np
import pandas as pd
import random
import math
import gensim
from scipy.special import psi, loggamma, polygamma
#from Telegram_chatbot import MTRobot
from Telegram_multi_chatbot import MTRobot
from sklearn.preprocessing import normalize
from module_GR_TensorLDA_BatchLearning import GR_TensorLDA
import time
# In[]:
def doc_generator_odt(user_tensor):
    docs = []
    list_docs_o = []
    list_docs_d = []
    list_docs_t = []
    wordcount = []
    for i in range(user_tensor.shape[0]):
        user = user_tensor[i,:,:,:]
        O, D, T = np.nonzero(user)
        count = user[np.nonzero(user)]
        
        wordcount.append(sum(count))
        
        doc = []
        for j in range(len(O)):
            word_o = 'O' + str(O[j]) #+ '_D' + str(D[j]) + '_T' + str(T[j])
            doc += int(count[j]) * [word_o]
        list_docs_o.append(doc)
        
        doc = []
        for k in range(len(D)):
            word_d = 'D' + str(D[k]) #+ '_D' + str(D[j]) + '_T' + str(T[j])
            doc += int(count[k]) * [word_d]
        list_docs_d.append(doc)
        
        doc = []
        for l in range(len(T)):
            word_t = 'T' + str(T[l]) #+ '_D' + str(D[j]) + '_T' + str(T[j])
            doc += int(count[l]) * [word_t]
        list_docs_t.append(doc)
        
    docs = pd.concat([pd.Series(list_docs_o), pd.Series(list_docs_d), pd.Series(list_docs_t), pd.Series(wordcount)], axis=1)
    docs.columns = ['O', 'D', 'T', 'wordcount']
    dictionary_o = gensim.corpora.Dictionary(docs['O'])
    dictionary_d = gensim.corpora.Dictionary(docs['D'])
    dictionary_t = gensim.corpora.Dictionary(docs['T'])
    #bow_corpus_o = [dictionary_o.doc2bow(doc) for doc in docs['O']]
    #bow_corpus_d = [dictionary_d.doc2bow(doc) for doc in docs['D']]
    #bow_corpus_t = [dictionary_t.doc2bow(doc) for doc in docs['T']]
    idx_corpus_o = [dictionary_o.doc2idx(doc) for doc in docs['O']]
    idx_corpus_d = [dictionary_d.doc2idx(doc) for doc in docs['D']]
    idx_corpus_t = [dictionary_t.doc2idx(doc) for doc in docs['T']]
    
    return docs, dictionary_o, dictionary_d, dictionary_t, idx_corpus_o, idx_corpus_d, idx_corpus_t
    

def perlexity(test_docs, idx_corpus, alpha, count_uz, beta):
    beta = np.exp(beta)
    num_topic = beta.shape[0]
    log_per = 0
    wordcount_sum = 0
    Kalpha = num_topic * alpha
    for u in range(len(test_docs)):
        theta = count_uz[u] / (len(test_docs.iloc[u]) + Kalpha)
        for w in range(len(test_docs.iloc[u])):
            log_per -= np.log( np.inner(beta[:,idx_corpus[u][w]], theta) + 1e-6 ) # phi[:,w]: R^(K*1)
        wordcount_sum += len(test_docs.iloc[u])
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
docs_arr = np.load('D:/Google Drive/HKUST-Office/Research/4th Work/Data/passenger data/docs_5k_jan_feb_arr.npy', allow_pickle=True)
docs = pd.DataFrame(data=docs_arr, columns=['TY', 'O', 'D', 'T', 'wordcount'])

#docs = docs_df[docs_df['wordcount'] >= 90]

#indx = list(range(len(docs_90)))
#random.shuffle(indx)
#docs = docs_90.iloc[indx[0:600]]


dictionary_o = gensim.corpora.Dictionary(docs['O'])
dictionary_d = gensim.corpora.Dictionary(docs['D'])
dictionary_t = gensim.corpora.Dictionary(docs['T'])

idx_corpus_o = [dictionary_o.doc2idx(doc) for doc in docs['O']] # corpus containing training and testing
idx_corpus_d = [dictionary_d.doc2idx(doc) for doc in docs['D']]
idx_corpus_t = [dictionary_t.doc2idx(doc) for doc in docs['T']]

# In[]
num_user = docs.shape[0]
num_station = max(len(dictionary_o), len(dictionary_d))
num_time = len(dictionary_t)

# In[1.2] Define global parameters and Initialization

# number of passengers for training
M = 76*64 #num_user
test_docs = docs.iloc[M:]
# number of topic
J = 10
K = 10
L = 4
# iteration times of variational EM algorithm
iterEM = 10
EM_CONVERGED = 0.001
EM_CONVERGED_fine_tune = 0.002
# iteration times of variational inference
iterInference = 20
VAR_CONVERGED = 0.0001

# initial value of hyperparameter alpha
alpha = 5
# sufficient statistic of alpha
alphaSS = 0
# the topic-word distribution (beta in D. Blei's paper)
betaO = np.zeros([J, num_station]) #+ 1e-5
betaD = np.zeros([K, num_station]) #+ 1e-5
betaT = np.zeros([L, num_time]) #+ 1e-5
# topic-word count, this is a sufficient statistic to calculate beta
count_zwo = np.zeros((J, num_station))     # sufficient statistic for beta^O
count_zwd = np.zeros((K, num_station))     # sufficient statistic for beta^D
count_zwt = np.zeros((L, num_time))        # sufficient statistic for beta^T
# topic count, sum of nzw with w ranging from [0, M-1], for calculating varphi
count_zo = np.zeros(J)
count_zd = np.zeros(K)
count_zt = np.zeros(L)

# inference parameter gamma
gamma = np.zeros((M, J, K, L))
# inference parameter phi
#phiO = np.zeros([maxItemNum(), J])
#phiD = np.zeros([maxItemNum(), K])
#phiT = np.zeros([maxItemNum(), L])

#likelihood = 0
#likelihood_old = 0

G_net = np.random.randint(2, size=(num_station, num_station))
G_poi = np.load('D:/Google Drive/HKUST-Office/Research/4th Work/Data/poi_sim.npy') 
# In[]
worker_idx = 3 -1
LAMBDA = [0.1]
mu = 0 # only POI graph on origin
nu = 0 # only POI graph on destination
save_dir = 'C:/Users/zlibn/Desktop/batch_5k_ascent'
# In[2] variational EM Algorithm
###############################################################################
#while (converged < 0 or converged > EM_CONVERGED or iteration<=2) and (i <= iterEM):
perplexity_matrix = np.zeros((len(LAMBDA),4))
likelihood_matrix = np.zeros((len(LAMBDA),2))

for lam_k, lam_v in enumerate(LAMBDA):
    
    #MTRobot.sendtext(worker_idx, " Start Lambda: {}".format(lam_v))
    print(f'Start Lambda: {lam_v}')
    
    time_0  = int(time.time()) 
    
    model = GR_TensorLDA(worker_idx, alpha, J, K, L, M, test_docs, iterEM, EM_CONVERGED, EM_CONVERGED_fine_tune, iterInference, VAR_CONVERGED)
    count_uzo, count_uzd, count_uzt, betaO, betaD, betaT, gamma, theta, likelihood, likelihood_evolu, perO_evolu, perD_evolu, perT_evolu \
    = model.fit(docs=docs, lam=lam_v, mu=mu, nu=nu, G_net=G_net, G_poi=G_poi, dictionary_o=dictionary_o, dictionary_d=dictionary_d, dictionary_t=dictionary_t, \
                idx_corpus_o=idx_corpus_o, idx_corpus_d=idx_corpus_d, idx_corpus_t=idx_corpus_t, num_user=num_user, num_station=num_station, num_time=num_time)
        
# In[]
    perO = perlexity(docs["O"].loc[0:M], idx_corpus_o, alpha, count_uzo, betaO)
    perD = perlexity(docs["D"].loc[0:M], idx_corpus_d, alpha, count_uzd, betaD)
    perT = perlexity(docs["T"].loc[0:M], idx_corpus_t, alpha, count_uzt, betaT)
    time_1  = int(time.time())
    
    perplexity_matrix[lam_k,:] = [lam_v, perO, perD, perT]
    #np.save(save_dir+'/perplexity'+'_lam'+f'_{lam_v}'+'_.npy', perplexity_matrix)
    
    likelihood_matrix[lam_k,:] = [lam_v, likelihood]
    #np.save(save_dir+'/likelihood'+'_lam'+f'_{lam_v}'+'_.npy', likelihood_matrix)

    np.save(save_dir+'/betaO'+'_lam'+f'_{lam_v}'+'_.npy', betaO)
    #np.save(save_dir+'/gradientO'+'_lam'+f'_{lam_v}'+'_.npy', gradientO)
    #np.save(save_dir+'/hessianO'+'_lam'+f'_{lam_v}'+'_.npy', hessianO)
    
    np.save(save_dir+'/betaD'+'_lam'+f'_{lam_v}'+'_.npy', betaD)
    np.save(save_dir+'/betaT'+'_lam'+f'_{lam_v}'+'_.npy', betaT)
    
    np.save(save_dir+'/gamma'+'_lam'+f'_{lam_v}'+'_.npy', gamma)
    np.save(save_dir+'/theta'+'_lam'+f'_{lam_v}'+'_.npy', theta)
    MTRobot.sendtext(worker_idx, f'saved all results for lambda {lam_v}')
    
    
# In[3] when the new test docs come

model_test = GR_TensorLDA(worker_idx, alpha, J, K, L, M, test_docs, iterEM, EM_CONVERGED, EM_CONVERGED_fine_tune, iterInference, VAR_CONVERGED)
count_uzo_t, count_uzd_t, count_uzt_t, likelihood_t = model_test.new_doc_infer(test_docs=test_docs, betaO=betaO, betaD=betaD, betaT=betaT, idx_corpus_o=idx_corpus_o, idx_corpus_d=idx_corpus_d, idx_corpus_t=idx_corpus_t)
perO = perlexity(test_docs["O"], idx_corpus_o[M:], alpha, count_uzo_t, betaO)
perD = perlexity(test_docs["D"], idx_corpus_d[M:], alpha, count_uzd_t, betaD)
perT = perlexity(test_docs["T"], idx_corpus_t[M:], alpha, count_uzt_t, betaT)

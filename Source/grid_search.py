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
    count_zwo, count_zo = initial_count(J, num_station)
    count_zwd, count_zd = initial_count(K, num_station)
    count_zwt, count_zt = initial_count(L, num_time)
    
    betaO = update_beta(count_zwo, count_zo)
    betaD = update_beta(count_zwd, count_zd)
    betaT = update_beta(count_zwt, count_zt)
    return count_zwo, count_zo, count_zwd, count_zd, count_zwt, count_zt, betaO, betaD, betaT


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

def d_blhood(beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi): # beta: R^(J * Vo), real value!

    d_beta_j = np.zeros(num_station)
    for w1 in range(num_station):
        d1 = lam * beta_no_g_exp[z, w1] / beta_old_exp[z, w1] - (1 - lam) * sum( (mu * G_net[w1, w2] + (1-mu) * G_poi[w1, w2]) * (beta_old_exp[z, w1]-beta_old_exp[z, w2]) for w2 in range(num_station))
        d_beta_j[w1] = d1
    return d_beta_j # R^(V * 1), real value!

def k_delta(i,j):
    if i == j:
        return 1
    else:
        return 0
    
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def d2_blhood(beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi): # beta: R^(J * Vo), real value!
    d2_beta = np.zeros((num_station, num_station))
    for w1 in range(num_station):
        for w2 in range(num_station):
            d2_beta[w1, w2] = -1 * (1 - lam) * (mu * G_net[w1, w2] + (1-mu) * G_poi[w1, w2]) - lam * k_delta(w1, w2) * beta_no_g_exp[z, w1] / ((beta_old_exp[z, w1])**2)
    if is_invertible(d2_beta) == False:
        d2_beta = d2_beta + 1e-6*np.random.rand(num_station, num_station) # add noise in hessian matrix to avoid singular matrix
    return d2_beta # R^(V * V), real value!
    
#Newton-Raphson Method will be applied for it
def update_beta_w_graph(lam, beta_no_g, mu, G_net, G_poi):
    
    num_topic = beta_no_g.shape[0]
    num_word = beta_no_g.shape[1]
    
    #count_zw, count_z = initial_count(num_topic, num_word)
    #beta_old = update_beta(count_zw, count_z)
    
    # !!!!!!!!!!!!!!!!!!!!!!!!
    beta_old_exp = np.exp(beta_no_g)
    beta_no_g_exp = np.exp(beta_no_g)
    
    #beta_w_g = np.exp(beta_old)
    beta_w_g_exp = np.zeros((num_topic, num_word)) # beta with graph, real value!
    
    #GRAD_THRESH = 0.001
    iteration = 0
    d_beta_j = [10] * num_word
    beta_w_g_exp_norm_old = 0.1
    #gradient_norm_old = 100
    converged = 10
    #converged_grad = 10
    while (converged > NT_CONVERGED):# and iteration < MAX_NT_ITER: #( converged_grad > GRAD_THRESH or  #all(math.fabs(df)> NEWTON_THRESH for df in d_beta_j) == True and 
        gradient = np.zeros((num_topic, num_word))
        hessian = np.zeros((num_topic, num_word, num_word))
        
        iteration += 1
        #MTRobot.sendtext(worker_idx, " -- Newton iter{}!".format(iteration))
        for z in range(num_topic):
            
            d_beta_j = d_blhood(beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi) # R^(V * 1), real value!
            gradient[z,:] = d_beta_j
            #MTRobot.sendtext(" ---- gradient calculated at topic{}!".format(z))
            
            #d2_beta = d2_blhood(beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi) # R^(V * V), real value!
            #hessian[z,:,:] = d2_beta#hessian.append(d2_beta)
            #MTRobot.sendtext(" ---- hessian calculated at topic{}!".format(z))
            
            beta_w_g_exp[z,:] = beta_old_exp[z,:] - g_step * d_beta_j  #np.dot(np.linalg.inv(d2_beta), d_beta_j) # real value! 
            #MTRobot.sendtext(" ---- beta with graph updated at topic{}!".format(z))
            
            #beta_old_exp[z,:] = beta_w_g_exp[z,:]
            
        #beta_old_exp = np.exp(beta_old)
        grad_np = sum(sum(gradient)) / np.fabs(sum(sum(gradient)))
        grad_scale = grad_np * sum(sum(np.fabs(gradient)))/(num_topic*num_word)
        beta_w_g_exp[beta_w_g_exp <= 0] = 1e-2 # to aviod non-feasible value
        beta_w_g_exp = normalize(beta_w_g_exp, norm='l1')
        beta_old_exp = beta_w_g_exp
        
        # Check convergence
        #gradient_norm = np.linalg.norm(gradient)
        beta_w_g_exp_norm = np.linalg.norm(beta_w_g_exp)
        
        converged = math.fabs(beta_w_g_exp_norm -beta_w_g_exp_norm_old) / beta_w_g_exp_norm_old
        #converged_grad = math.fabs(gradient_norm -gradient_norm_old) / gradient_norm_old
        beta_w_g_exp_norm_old = beta_w_g_exp_norm
        #gradient_norm_old = gradient_norm
        
        #print(f'beta: {beta_w_g_norm:.3f}  gradient_scale:{grad_scale:.3f}  Converged: {converged:.3f}')
        #MTRobot.sendtext(worker_idx, f' Newton iter{iteration}  gradient:{grad_scale:.5f} beta: {beta_w_g_exp_norm:.5f} betacon: {converged:.5f}') 
        print(' Newton iter{iteration}  gradient:{grad_scale:.5f} beta: {beta_w_g_exp_norm:.5f} betacon: {converged:.5f}')
    
    beta_w_g = np.log(beta_w_g_exp)
    return beta_w_g, gradient, hessian # log value

def converge_paras(paraO_norm, paraD_norm, paraT_norm, paraO_norm_old, paraD_norm_old, paraT_norm_old, PARA_CONVERGED):
    if math.fabs(paraO_norm -paraO_norm_old) / paraO_norm_old < PARA_CONVERGED and math.fabs(paraD_norm -paraD_norm_old) / paraD_norm_old < PARA_CONVERGED and math.fabs(paraT_norm -paraT_norm_old) / paraT_norm_old < PARA_CONVERGED:
        return True
    else:
        return False

# update variational parameters : gamma, phiO, phiD, phiT
# doc: DataFrame
def variationalInference(docs, u, gamma, phiO, phiD, phiT):
    converged = 1
    i_infer = 0
    phisumO = 0
    phisumD = 0
    phisumT = 0
    bool_phi_converge = False
    phiO_norm_old = 0.1
    phiD_norm_old = 0.1
    phiT_norm_old = 0.1
    likelihood_u = 0
    likelihood_u_old = 0.1
    oldphiO = np.zeros(J)
    oldphiD = np.zeros(K)
    oldphiT = np.zeros(L)
    digamma_gamma = np.zeros((J, K, L))
     
    # Initialization for phiO, phiD, phiT:
    for j in range(J):
        for wo in range(len(idx_corpus_o[u])): # number of (unigue) word
            phiO[wo, j] = 1.0 / J
    for k in range(K):
        for wd in range(len(idx_corpus_d[u])): # number of unigue word
            phiD[wd, k] = 1.0 / K
    for l in range(L):
        for wt in range(len(idx_corpus_t[u])): # number of unigue word
            phiT[wt, l] = 1.0 / L
    # Initialization for gamma
    for j in range(J):
        for k in range(K):
            for l in range(L):
                gamma[u, j, k, l] = alpha + docs.iloc[u]['wordcount'] * 1.0 / (J*K*L)
                digamma_gamma[j, k, l] = psi(gamma[u, j, k, l])
    
    while (converged > VAR_CONVERGED  or bool_phi_converge==False) and i_infer <= iterInference:
    #for i_infer in range(iterInference):
        #MTRobot.sendtext("---Variational Inference Iter {}".format(i_infer))
        # To update phiO:
        for wo in range(len(idx_corpus_o[u])):
            phisumO = 0
            for j in range(J):
                oldphiO[j] = phiO[wo, j]
                phiO[wo, j] = sum(math.exp(oldphiD[k]) * math.exp(oldphiT[l]) * digamma_gamma[j, k, l] for k in range(K) for l in range(L)) + betaO[j, idx_corpus_o[u][wo]] # bow_corpus_o[u][wo][0] # docs[d].itemIdList[wo]
                if j > 0:
                    phisumO = math.log(math.exp(phisumO) + math.exp(phiO[wo, j]))
                else:
                    phisumO = phiO[wo, j]
            for j in range(J):
                phiO[wo, j] = math.exp(phiO[wo, j] - phisumO) # normalization
                # Output: Real_phiO
        phiO_norm = np.linalg.norm(phiO)

        # To update phiD:
        #MTRobot.sendtext("Update phiD: iter {}".format(iteration))
        for wd in range(len(idx_corpus_d[u])):
            phisumD = 0
            for k in range(K):
                oldphiD[k] = phiD[wd, k]
                phiD[wd, k] = sum(math.exp(oldphiO[j]) * math.exp(oldphiT[l]) * digamma_gamma[j, k, l] for j in range(J) for l in range(L)) + betaD[k, idx_corpus_d[u][wd]] # docs[d].itemIdList[wo]
                if k > 0:
                    phisumD = math.log(math.exp(phisumD) + math.exp(phiD[wd, k]))
                else:
                    phisumD = phiD[wd, k]
            for k in range(K):
                phiD[wd, k] = math.exp(phiD[wd, k] - phisumD) # normalization
                # Output: Real_phiD
        phiD_norm = np.linalg.norm(phiD)

        # To update phiT:
        #MTRobot.sendtext("Update phiT: iter {}".format(iteration))
        for wt in range(len(idx_corpus_t[u])):
            phisumT = 0
            for l in range(L):
                oldphiT[l] = phiT[wt, l]
                phiT[wt, l] = sum(math.exp(oldphiO[j]) * math.exp(oldphiD[k]) * digamma_gamma[j, k, l] for j in range(J) for k in range(K)) + betaT[l, idx_corpus_t[u][wt]] # docs[d].itemIdList[wo]
                if l > 0:
                    phisumT = math.log(math.exp(phisumT) + math.exp(phiT[wt, l]))
                else:
                    phisumT = phiT[wt, l]
            for l in range(L):
                phiT[wt, l] = math.exp(phiT[wt, l] - phisumT) # normalization over topic dimension
                # Output: Real_phiT
        phiT_norm = np.linalg.norm(phiT)
                
        # To updata gamma:
        #MTRobot.sendtext("Update gamma: iter {}".format(iteration))
        gammaSum = 0
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    gamma[u, j, k, l] = alpha + sum( phiO[w, j] * phiD[w, k] * phiT[w, l] for w in range(int(docs.iloc[u]['wordcount'])))
                    digamma_gamma[j, k, l] = psi(gamma[u, j, k, l])
                    gammaSum += gamma[u, j, k, l]
         
        #MTRobot.sendtext(f'calculate Likelihood for iteration {iTer}') 
        likelihood_u = compute_likelihood(u, digamma_gamma, gammaSum, phiO, phiD, phiT, betaO, betaD, betaT)
        converged = (likelihood_u_old - likelihood_u) / likelihood_u_old
        likelihood_u_old = likelihood_u
        
        bool_phi_converge = converge_paras(phiO_norm, phiD_norm, phiT_norm, phiO_norm_old, phiD_norm_old, phiT_norm_old, PARA_CONVERGED=0.0005) #phi_norm magnitude: 30
        
        phiO_norm_old = phiO_norm
        phiD_norm_old = phiD_norm
        phiT_norm_old = phiT_norm
        
        #MTRobot.sendtext(worker_idx, f'User {u} -- Likelihood: {likelihood_u:.5f}   Converged: {converged:.5f}')
        #MTRobot.sendtext(worker_idx, f'phiO: {phiO_norm:.5f}   phiD: {phiD_norm:.5f}   phiT: {phiT_norm:.5f}') 
        
        i_infer = i_infer + 1
    #MTRobot.sendtext(worker_idx, f'User {u} -- Likelihood: {likelihood_u:.5f}   Converged: {converged:.5f}')
    return phiO, phiD, phiT, gamma, likelihood_u

def compute_likelihood(u, digamma_gamma, gammaSum, phiO, phiD, phiT, betaO, betaD, betaT):
    likelihood = 0
    digsum = psi(gammaSum) 
    likelihood = loggamma(alpha*J*K*L) - J*K*L*loggamma(alpha) - (loggamma(gammaSum)) # 1.1， 1.2， 1.3

    for j in range(J):
        for k in range(K):
            for l in range(L):
                likelihood += (alpha-1)*(digamma_gamma[j,k,l]-digsum) + loggamma(gamma[u,j,k,l]) - (gamma[u,j,k,l]-1)*(digamma_gamma[j,k,l]-digsum) # 2.1， 2.2， 2.3
                for w in range(int(docs.iloc[u]['wordcount'])):
                    if phiO[w,j]>0 and phiD[w,k]>0 and phiT[w,l]>0:
                        likelihood += phiO[w, j] * phiD[w, k] * phiT[w, l] * (digamma_gamma[j,k,l]-digsum) # 3.1
    for j in range(J):
        for wo in range(len(idx_corpus_o[u])):
            likelihood += - phiO[wo, j] * math.log(phiO[wo, j]) + phiO[wo, j] * betaO[j, idx_corpus_o[u][wo]] # 3.2 O; 3.3 O
    for k in range(K):
        for wd in range(len(idx_corpus_d[u])):
            likelihood += - phiD[wd, k] * math.log(phiD[wd, k]) + phiD[wd, k] * betaD[k, idx_corpus_d[u][wd]] # 3.2 D; 3.3 D
    for l in range(L):
        for wt in range(len(idx_corpus_t[u])):
            likelihood += - phiT[wt, l] * math.log(phiT[wt, l]) + phiT[wt, l] * betaT[l, idx_corpus_t[u][wt]] # 3.2 T; 3.3 T
    return likelihood
    

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
docs_df = pd.DataFrame(data=docs_arr, columns=['TY', 'O', 'D', 'T', 'wordcount'])

docs = docs_df[docs_df['wordcount'] >= 90]

#indx = list(range(len(docs_90)))
#random.shuffle(indx)
#docs = docs_90.iloc[indx[0:600]]


dictionary_o = gensim.corpora.Dictionary(docs['O'])
dictionary_d = gensim.corpora.Dictionary(docs['D'])
dictionary_t = gensim.corpora.Dictionary(docs['T'])

idx_corpus_o = [dictionary_o.doc2idx(doc) for doc in docs['O']]
idx_corpus_d = [dictionary_d.doc2idx(doc) for doc in docs['D']]
idx_corpus_t = [dictionary_t.doc2idx(doc) for doc in docs['T']]

# In[]
num_user = docs.shape[0]
num_station = max(len(dictionary_o), len(dictionary_d))
num_time = len(dictionary_t)

# In[1.2] Define global parameters and Initialization

# number of passengers for training
M = 300 #num_user
# number of topic
J = 10
K = 10
L = 4
# iteration times of variational EM algorithm
iterEM = 20
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
phiO = np.zeros([maxItemNum(), J])
phiD = np.zeros([maxItemNum(), K])
phiT = np.zeros([maxItemNum(), L])

#likelihood = 0
#likelihood_old = 0

G_net = np.random.randint(2, size=(num_station, num_station))
G_poi = np.load('D:/Google Drive/HKUST-Office/Research/4th Work/Data/poi_sim.npy') 
# In[]
worker_idx = 4 -1
LAMBDA = [0.2, 0.1]
mu = 0 # only POI graph on origin
#nu = 0 # only POI graph on destination
save_dir = 'C:/Users/zlibn/Desktop/save_results'
# In[2] variational EM Algorithm
###############################################################################
#while (converged < 0 or converged > EM_CONVERGED or iteration<=2) and (i <= iterEM):
perplexity_matrix = np.zeros((len(LAMBDA),4))
likelihood_matrix = np.zeros((len(LAMBDA),2))

for lam_k, lam_v in enumerate(LAMBDA):
    
    #MTRobot.sendtext(worker_idx, " Start Lambda: {}".format(lam_v))
    print(f'Start Lambda: {lam_v}')
    
    # initial so that after this for loop program can still enter next while loop
    i_em = 0
    betaO_norm_old = 0.1
    betaD_norm_old = 0.1
    betaT_norm_old = 0.1
    bool_beta_converge = False
    converged = -1
    likelihood_old = 0.1
    # initialization of the model parameter varphi, the update of alpha is ommited
    count_zwo, count_zo, count_zwd, count_zd, count_zwt, count_zt, betaO, betaD, betaT = initialLdaModel()

    
    while (converged < 0 or converged > EM_CONVERGED or i_em <2 or bool_beta_converge == False) and i_em <= iterEM: #
    #    iteration += 1
    #for i_em in range(iterEM):
        likelihood = 0
        #MTRobot.sendtext(worker_idx, " -- Start EM interation: {}".format(i_em))
        print(f'-- Start EM interation: {i_em}')
        count_zwo = np.zeros((J, num_station))     # sufficient statistic for beta^O
        count_zwd = np.zeros((K, num_station))     # sufficient statistic for beta^D
        count_zwt = np.zeros((L, num_time))        # sufficient statistic for beta^T
        count_zo = np.zeros(J)
        count_zd = np.zeros(K)
        count_zt = np.zeros(L)
        count_uzo = np.zeros((M, J))
        count_uzd = np.zeros((M, K))
        count_uzt = np.zeros((M, L))
        theta = np.zeros((M, J, K, L))
        alphaSS = 0
        
        # iteration times of newton method # varies for each EM iteration because of 
        # MAX_NT_ITER = 20#50 # 10
        NT_CONVERGED = 0.001 # since beta_w_g_norm magnitude is 0.7
        g_step = 0.001

        # E-Step
        #print("-start variational Inference E-step")
        #MTRobot.sendtext(worker_idx, " ---- start variational Inference E-step")
        print(" ---- start variational Inference E-step")
        for u in range(M):
            #MTRobot.sendtext("------Passenger{}".format(u))
            phiO, phiD, phiT, gamma, likelihood_u = variationalInference(docs, u, gamma, phiO, phiD, phiT)
            likelihood += likelihood_u
            #converged = (likelihood_old - likelihood) / (likelihood_old);
            gammaSum = 0
            for j in range(J):
                for k in range(K):
                    for l in range(L):
                        gammaSum += gamma[u, j, k, l]
                        alphaSS += psi(gamma[u, j, k, l])
            alphaSS -= J * K * L * psi(gammaSum)
            
            # To update count_zwo, count_zo
            for wo in range(len(idx_corpus_o[u])):
                for j in range(J):
                    count_zwo[j, idx_corpus_o[u][wo]] += phiO[wo, j] # count_zwo[j, bow_corpus_o[u][wo][0]] += bow_corpus_o[u][wo][1] * phiO[wo, j] # nzw[z][docs[d].itemIdList[w]] += docs[d].itemCountList[w] * phi[w, z]
                    count_zo[j] += phiO[wo, j] # nz[z] += docs[d].itemCountList[w] * phi[w, z]
                    count_uzo[u, j] += phiO[wo, j]
            
            # To update count_zwd, count_zd
            for wd in range(len(idx_corpus_d[u])):
                for k in range(K):
                    count_zwd[k, idx_corpus_d[u][wd]] += phiD[wd, k] 
                    count_zd[k] += phiD[wd, k]
                    count_uzd[u, k] += phiD[wd, k]
            
            # To update count_zwo, count_zo
            for wt in range(len(idx_corpus_t[u])):
                for l in range(L):
                    count_zwt[l, idx_corpus_t[u][wt]] += phiT[wt, l] 
                    count_zt[l] += phiT[wt, l]
                    count_uzt[u, l] += phiT[wt, l]
            
            # To update theta_u
            for j in range(J):
                for k in range(K):
                    for l in range(L):
                        theta[u, j, k, l] = sum( phiO[w, j] * phiD[w, k] * phiT[w, l] for w in range(int(docs.iloc[u]['wordcount'])) )
            theta[u, :, :, :] = theta[u, :, :, :] / sum(sum(sum(theta[u, :, :, :])))
            #theta[u, :, :, :] = gamma[u, :, :, :] / sum(sum(sum(gamma[u, :, :, :])))
        
        converged = (likelihood_old - likelihood) / (likelihood_old)
        likelihood_old = likelihood
        # M-Step
        print("---- start variational Inference M-step")
        #MTRobot.sendtext(worker_idx, " ---- start variational Inference M-step")
        
        # Update betaO
        #betaO = update_beta(count_zwo, count_zo)
        
        MTRobot.sendtext(worker_idx, " ------ Origin ")
        if converged < EM_CONVERGED_fine_tune and converged > 0: # start fine tune for beta when EM algorithm stabilizes
            # MAX_NT_ITER = 2 * MAX_NT_ITER
            NT_CONVERGED = 0.5 * NT_CONVERGED
        betaO_no_g = update_beta(count_zwo, count_zo) # betaO, gradient, hessian = update_beta_w_graph(lam, count_zwo, count_zo, mu, G_net, G_poi) # update_beta(count_zwo, count_zo) 
        betaO, gradientO, hessianO = update_beta_w_graph(lam_v, betaO_no_g, mu, G_net, G_poi)
        #MTRobot.sendtext(worker_idx, " ------ End Origin ")
        
        # Update betaD
        betaD = update_beta(count_zwd, count_zd)
        
        # Update betaT
        betaT = update_beta(count_zwt, count_zt)
        
        betaO_norm = np.linalg.norm(np.exp(betaO))
        betaD_norm = np.linalg.norm(np.exp(betaD))
        betaT_norm = np.linalg.norm(np.exp(betaT))
        
        # check for convergence
        bool_beta_converge = converge_paras(betaO_norm, betaD_norm, betaT_norm, betaO_norm_old, betaD_norm_old, betaT_norm_old, PARA_CONVERGED=0.0015) # beta_norm magnitude: 0.7
        
        # update old parameters for next EM-iteration
        betaO_norm_old = betaO_norm
        betaD_norm_old = betaD_norm
        betaT_norm_old = betaT_norm
        
        #MTRobot.sendtext(worker_idx, f'End EM Iter {i_em} -- Likelihood: {likelihood:.5f}   Converged: {converged:.5f}')
        #MTRobot.sendtext(worker_idx, f'betaO: {betaO_norm:.5f}   betaD: {betaD_norm:.5f}   betaT: {betaT_norm:.5f}') 
        print(f'End EM Iter {i_em} -- Likelihood: {likelihood:.5f}   Converged: {converged:.5f}')
        print(f'betaO: {betaO_norm:.5f}   betaD: {betaD_norm:.5f}   betaT: {betaT_norm:.5f}')
        
        i_em = i_em +1
        
    perO = perlexity(docs["O"].loc[0:300], idx_corpus_o, alpha, count_uzo, betaO)
    perD = perlexity(docs["D"].loc[0:300], idx_corpus_d, alpha, count_uzd, betaD)
    perT = perlexity(docs["T"].loc[0:300], idx_corpus_t, alpha, count_uzt, betaT)
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
    
    



    

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:12:14 2020

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
#from Telegram_chatbot_2 import MTRobot
from sklearn.preprocessing import normalize
import time

class GR_OnlineTensorLDA:
    def __init__(self, worker_idx, alpha, num_station, num_time, J, K, L, M, S, iterEM, EM_CONVERGED, EM_CONVERGED_fine_tune, iterInference, VAR_CONVERGED, tau0, kappa, _updatect):
        """
        Explanation:
        worker_idx: telegram chatbox id selection;
        lam: lambda as the graph regularization tuning parameter
        mu: relative effect of two graphs for origin dimension
        nu: relative effect of two graphs for destination dimension
        """
        self.worker_idx = worker_idx
        #self.lam = lam
        #self.mu = mu
        #self.nu = nu
        self.alpha = alpha
        
        self.num_station = num_station
        self.num_time = num_time
        
        self.J = J
        self.K = K
        self.L = L
        self.M = M
        self.S = S
        self.num_batch = int(self.M/self.S)
        self.iterEM = iterEM
        self.EM_CONVERGED = EM_CONVERGED
        self.EM_CONVERGED_fine_tune = EM_CONVERGED_fine_tune
        self.iterInference = iterInference
        self.VAR_CONVERGED = VAR_CONVERGED
        
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = _updatect
        
        # self.betaO_o = betaO_o
        # self.betaD_o = betaD_o
        # self.betaT_o = betaT_o
        self.theta = np.zeros((self.M, J, K, L))
        # inference parameter gamma
        self.gamma = np.zeros((self.M, J, K, L)) 
        # gamma are supposed to be a corpus-wide variable, so gamma = np.zeros((M, J, K, L)) but since gamma is not importatn, so we just gamma = np.zeros((S, J, K, L))
        
        #self.betaO_o = np.zeros([self.J, self.num_station]) #+ 1e-5
        #self.betaD_o = np.zeros([self.K, self.num_station]) #+ 1e-5
        #self.betaT_o = np.zeros([self.L, self.num_time]) #+ 1e-5
        
    def maxItemNum(self, M, docs):
        num = 0
        for u in range(M):
            if  docs.iloc[u]['wordcount'] > num: #len(docs[d].itemIdList): number of unique words in a document
                num = int(docs.iloc[u]['wordcount'])
        return num
        
    def initial_count(self, num_topic, num_word):
        count_zw = np.zeros((num_topic, num_word))     # sufficient statistic for beta
        count_z = np.zeros(num_topic)
        for z in range(num_topic):
            for w in range(num_word):
                count_zw[z, w] += 1.0/num_word + random.random() 
                count_z[z] += count_zw[z, w]
        return count_zw, count_z
        
    def initialLdaModel(self, num_batch, num_station, num_time):
        
        count_zwo, count_zo = self.initial_count(self.J, num_station) # defined as global parameter, used to calculate betaO
        count_zwd, count_zd = self.initial_count(self.K, num_station)
        count_zwt, count_zt = self.initial_count(self.L, num_time)
        
        betaO = self.update_beta(num_batch, count_zwo, count_zo)
        betaD = self.update_beta(num_batch, count_zwd, count_zd)
        betaT = self.update_beta(num_batch, count_zwt, count_zt)
        return count_zwo, count_zo, count_zwd, count_zd, count_zwt, count_zt, betaO, betaD, betaT

    # update model parameters : beta (the topic-word parameter， (real-value) log-value is actually calculated here)
    # (the update of alpha is ommited)
    def update_beta(self, num_batch, count_zw, count_z):
        num_topic = count_zw.shape[0]
        num_word = count_zw.shape[1]
        beta = np.zeros((num_topic, num_word))
    
        for z in range(num_topic):
            for w in range(0, num_word):
                if(count_zw[z, w] > 0):
                    beta[z, w] = math.log(num_batch) + math.log(count_zw[z, w]) - math.log(count_z[z]) # beta[z, w] = count_zw[z, w] / count_z[z] 
                else:
                    beta[z, w] = -100 # beta[z, w] = 0
        return beta
    
    def d_blhood(self, num_batch, num_station, beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi): # beta: R^(J * Vo), real value!
    
        d_beta_j = np.zeros(num_station)
        for w1 in range(num_station):
            d1 = lam * num_batch * beta_no_g_exp[z, w1] / beta_old_exp[z, w1] - (1 - lam) * sum( (mu * G_net[w1, w2] + (1-mu) * G_poi[w1, w2]) * (beta_old_exp[z, w1]-beta_old_exp[z, w2]) for w2 in range(num_station))
            d_beta_j[w1] = d1
        return d_beta_j # R^(V * 1), real value!
    
    def k_delta(self, i,j):
        if i == j:
            return 1
        else:
            return 0
        
    def is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
    
    def d2_blhood(self, num_station, beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi): # beta: R^(J * Vo), real value!
        d2_beta = np.zeros((num_station, num_station))
        for w1 in range(num_station):
            for w2 in range(num_station):
                d2_beta[w1, w2] = -1 * (1 - lam) * (mu * G_net[w1, w2] + (1-mu) * G_poi[w1, w2]) - lam * self.k_delta(w1, w2) * beta_no_g_exp[z, w1] / ((beta_old_exp[z, w1])**2)
        if self.is_invertible(d2_beta) == False:
            d2_beta = d2_beta + 1e-6*np.random.rand(num_station, num_station) # add noise in hessian matrix to avoid singular matrix
        return d2_beta # R^(V * V), real value!
        
    #Newton-Raphson Method will be applied for it
    def update_beta_w_graph(self, num_batch, num_station, lam, beta_no_g, mu, G_net, G_poi, NT_CONVERGED, g_step):
        
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
                
                d_beta_j = self.d_blhood(num_batch, num_station, beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi) # R^(V * 1), real value!
                gradient[z,:] = d_beta_j
                #MTRobot.sendtext(" ---- gradient calculated at topic{}!".format(z))
                
                #d2_beta = self.d2_blhood(num_station, beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi) # R^(V * V), real value!
                #hessian[z,:,:] = d2_beta#hessian.append(d2_beta)
                #MTRobot.sendtext(" ---- hessian calculated at topic{}!".format(z))
                
                beta_w_g_exp[z,:] = beta_old_exp[z,:] + g_step * d_beta_j  # we are maximizing, so it's gradient ascend # - np.dot(np.linalg.inv(d2_beta), d_beta_j) # with hession is minus since hessian matrix is negative definite for concave # real value! 
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
            #MTRobot.sendtext(self.worker_idx, f' Newton iter{iteration}  gradient:{grad_scale:.5f} beta: {beta_w_g_exp_norm:.5f} betacon: {converged:.5f}') #!!!
            #print(f'Newton iter{iteration}  gradient:{grad_scale:.5f} beta: {beta_w_g_exp_norm:.5f} betacon: {converged:.5f}')
        
        beta_w_g = np.log(beta_w_g_exp)
        return beta_w_g, gradient, hessian # log value
    
    def converge_paras(self, paraO_norm, paraD_norm, paraT_norm, paraO_norm_old, paraD_norm_old, paraT_norm_old, PARA_CONVERGED):
        if math.fabs(paraO_norm -paraO_norm_old) / paraO_norm_old < PARA_CONVERGED and math.fabs(paraD_norm -paraD_norm_old) / paraD_norm_old < PARA_CONVERGED and math.fabs(paraT_norm -paraT_norm_old) / paraT_norm_old < PARA_CONVERGED:
            return True
        else:
            return False
    
    # update variational parameters : gamma, phiO, phiD, phiT
    # doc: DataFrame
    def variationalInference(self, i, docs_minibatch, s, u, gamma, phiO, phiD, phiT, betaO, betaD, betaT, idx_corpus_o, idx_corpus_d, idx_corpus_t):
        """
        Here the docs will be fed with docs_minibatch: s:[0,S); u[i*S, (i+1)*S)
        """
        
        J = self.J
        K = self.K
        L = self.L
        S = self.S
        alpha = self.alpha
        #u = i*S+s
        
        converged = 1
        i_infer = 0
        phisumO = 0
        phisumD = 0
        phisumT = 0
        bool_phi_converge = False
        phiO_norm_old = 0.1
        phiD_norm_old = 0.1
        phiT_norm_old = 0.1
        likelihood_s = 0
        likelihood_s_old = 0.1
        oldphiO = np.zeros(self.J)
        oldphiD = np.zeros(K)
        oldphiT = np.zeros(L)
        digamma_gamma = np.zeros((J, K, L))
        
        #""" variational  """"
        # Initialization for phiO, phiD, phiT:
        for j in range(J):
            for wo in range(len(idx_corpus_o[u])): # idx_corpus_o[u] used u since idx_corpus_o is a corpus-wide input # number of (unigue) word
                phiO[wo, j] = 1.0 / J 
        for k in range(K):
            for wd in range(len(idx_corpus_d[u])): # number of unigue word
                phiD[wd, k] = 1.0 / K
        for l in range(L):
            for wt in range(len(idx_corpus_t[u])): # number of unigue word
                phiT[wt, l] = 1.0 / L
        # Initialization for gamma
        for j in range(self.J):
            for k in range(self.K):
                for l in range(self.L):
                    gamma[u, j, k, l] = alpha + docs_minibatch.iloc[s]['wordcount'] * 1.0 / (self.J * self.K * self.L)  # s (mini-batch) is mapping to u (corpus-wide) by u = i*S+s
                    # originally we need a mapping from s to u, since gamma should be a corpus-wide parameter
                    # , but gamma is not important, so we just gamma[s, j, k, l]
                    digamma_gamma[j, k, l] = psi(gamma[u, j, k, l])
        
        while (converged > self.VAR_CONVERGED  or bool_phi_converge==False) and i_infer <= self.iterInference:
        #for i_infer in range(iterInference):
            #MTRobot.sendtext(self.worker_idx,"---Variational Inference Iter {}".format(i_infer))
            # To update phiO:
            for wo in range(len(idx_corpus_o[u])):
                phisumO = 0
                for j in range(self.J):
                    oldphiO[j] = phiO[wo, j]
                    phiO[wo, j] = sum(math.exp(oldphiD[k]) * math.exp(oldphiT[l]) * digamma_gamma[j, k, l] for k in range(self.K) for l in range(self.L)) + betaO[j, idx_corpus_o[u][wo]] # bow_corpus_o[u][wo][0] # docs[d].itemIdList[wo]
                    if j > 0:
                        phisumO = math.log(math.exp(phisumO) + math.exp(phiO[wo, j]))
                    else:
                        phisumO = phiO[wo, j]
                for j in range(self.J):
                    phiO[wo, j] = math.exp(phiO[wo, j] - phisumO) # normalization
                    # Output: Real_phiO
            phiO_norm = np.linalg.norm(phiO)
    
            # To update phiD:
            #MTRobot.sendtext("Update phiD: iter {}".format(iteration))
            for wd in range(len(idx_corpus_d[u])):
                phisumD = 0
                for k in range(self.K):
                    oldphiD[k] = phiD[wd, k]
                    phiD[wd, k] = sum(math.exp(oldphiO[j]) * math.exp(oldphiT[l]) * digamma_gamma[j, k, l] for j in range(self.J) for l in range(self.L)) + betaD[k, idx_corpus_d[u][wd]] # docs[d].itemIdList[wo]
                    if k > 0:
                        phisumD = math.log(math.exp(phisumD) + math.exp(phiD[wd, k]))
                    else:
                        phisumD = phiD[wd, k]
                for k in range(self.K):
                    phiD[wd, k] = math.exp(phiD[wd, k] - phisumD) # normalization
                    # Output: Real_phiD
            phiD_norm = np.linalg.norm(phiD)
    
            # To update phiT:
            #MTRobot.sendtext("Update phiT: iter {}".format(iteration))
            for wt in range(len(idx_corpus_t[u])):
                phisumT = 0
                for l in range(self.L):
                    oldphiT[l] = phiT[wt, l]
                    phiT[wt, l] = sum(math.exp(oldphiO[j]) * math.exp(oldphiD[k]) * digamma_gamma[j, k, l] for j in range(self.J) for k in range(self.K)) + betaT[l, idx_corpus_t[u][wt]] # docs[d].itemIdList[wo]
                    if l > 0:
                        phisumT = math.log(math.exp(phisumT) + math.exp(phiT[wt, l]))
                    else:
                        phisumT = phiT[wt, l]
                for l in range(self.L):
                    phiT[wt, l] = math.exp(phiT[wt, l] - phisumT) # normalization over topic dimension
                    # Output: Real_phiT
            phiT_norm = np.linalg.norm(phiT)
                    
            # To updata gamma:
            #MTRobot.sendtext("Update gamma: iter {}".format(iteration))
            #
            gammaSum = 0
            for j in range(self.J):
                for k in range(self.K):
                    for l in range(self.L):
                        gamma[u, j, k, l] = alpha + sum( phiO[w, j] * phiD[w, k] * phiT[w, l] for w in range(int(docs_minibatch.iloc[s]['wordcount'])))
                        digamma_gamma[j, k, l] = psi(gamma[u, j, k, l])
                        gammaSum += gamma[u, j, k, l]
             
            #MTRobot.sendtext(f'calculate Likelihood for iteration {iTer}') 
            likelihood_s = self.compute_likelihood(docs_minibatch, s, u, gamma, digamma_gamma, gammaSum, phiO, phiD, phiT, betaO, betaD, betaT, idx_corpus_o, idx_corpus_d, idx_corpus_t)
            converged = (likelihood_s_old - likelihood_s) / likelihood_s_old
            likelihood_s_old = likelihood_s
            
            bool_phi_converge = self.converge_paras(phiO_norm, phiD_norm, phiT_norm, phiO_norm_old, phiD_norm_old, phiT_norm_old, PARA_CONVERGED=0.0005) #phi_norm magnitude: 30
            
            phiO_norm_old = phiO_norm
            phiD_norm_old = phiD_norm
            phiT_norm_old = phiT_norm
            
            #MTRobot.sendtext(worker_idx, f'User {u} -- Likelihood: {likelihood_u:.5f}   Converged: {converged:.5f}')
            #MTRobot.sendtext(worker_idx, f'phiO: {phiO_norm:.5f}   phiD: {phiD_norm:.5f}   phiT: {phiT_norm:.5f}') 
            
            i_infer = i_infer + 1
        #MTRobot.sendtext(worker_idx, f'User {u} -- Likelihood: {likelihood_u:.5f}   Converged: {converged:.5f}')
        return phiO, phiD, phiT, gamma, likelihood_s
    
    def compute_likelihood(self, docs_minibatch, s, u, gamma, digamma_gamma, gammaSum, phiO, phiD, phiT, betaO, betaD, betaT, idx_corpus_o, idx_corpus_d, idx_corpus_t):
        J = self.J
        K = self.K
        L = self.L
        alpha = self.alpha
        
        likelihood = 0
        digsum = psi(gammaSum) 
        likelihood = loggamma(alpha*J *K *L) - J * K * L * loggamma(alpha) - (loggamma(gammaSum)) # 1.1， 1.2， 1.3
    
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    likelihood += (alpha-1)*(digamma_gamma[j,k,l]-digsum) + loggamma(gamma[u,j,k,l]) - (gamma[u,j,k,l]-1)*(digamma_gamma[j,k,l]-digsum) # 2.1， 2.2， 2.3
                    for w in range(int(docs_minibatch.iloc[s]['wordcount'])):
                        if phiO[w,j]>0 and phiD[w,k]>0 and phiT[w,l]>0:
                            likelihood += phiO[w, j] * phiD[w, k] * phiT[w, l] * (digamma_gamma[j,k,l]-digsum) # 3.1
        for j in range(self.J):
            for wo in range(len(idx_corpus_o[u])):
                if phiO[wo,j]>0:
                    likelihood += - phiO[wo, j] * math.log(phiO[wo, j]) + phiO[wo, j] * betaO[j, idx_corpus_o[u][wo]] # 3.2 O; 3.3 O
        for k in range(self.K):
            for wd in range(len(idx_corpus_d[u])):
                if phiD[wd,k]>0:
                    likelihood += - phiD[wd, k] * math.log(phiD[wd, k]) + phiD[wd, k] * betaD[k, idx_corpus_d[u][wd]] # 3.2 D; 3.3 D
        for l in range(self.L):
            for wt in range(len(idx_corpus_t[u])):
                if phiT[wt,l]>0:
                    likelihood += - phiT[wt, l] * math.log(phiT[wt, l]) + phiT[wt, l] * betaT[l, idx_corpus_t[u][wt]] # 3.2 T; 3.3 T
        return likelihood
    
    def dict_corpus(self, docs):
        # To get the dictionary and corpus on each dimension
        dictionary_o = gensim.corpora.Dictionary(docs['O'])
        dictionary_d = gensim.corpora.Dictionary(docs['D'])
        dictionary_t = gensim.corpora.Dictionary(docs['T'])
        
        idx_corpus_o = [dictionary_o.doc2idx(doc) for doc in docs['O']]
        idx_corpus_d = [dictionary_d.doc2idx(doc) for doc in docs['D']]
        idx_corpus_t = [dictionary_t.doc2idx(doc) for doc in docs['T']]
        
        # To get the size of vacabulary
        num_user = docs.shape[0]
        num_station = max(len(dictionary_o), len(dictionary_d))
        num_time = len(dictionary_t)
        return dictionary_o, dictionary_d, dictionary_t, idx_corpus_o, idx_corpus_d, idx_corpus_t, num_user, num_station, num_time
        
        
    def fit(self, i, docs_minibatch, lam, mu, nu, G_net, G_poi, dictionary_o, dictionary_d, dictionary_t, idx_corpus_o, idx_corpus_d, idx_corpus_t, num_user, num_station, num_time, betaO, betaD, betaT):
        """ 
        the docs here is mini-batch docs as a moving window;
        and the fit function here is to fit learn each mini-batch;
        betaO, betaD, betaT: here are parameters learned from each mini-batch, with assuming the entire corpus at this moment is this mini-batch repeated num_batch times
        """
        i = i
        J = self.J
        K = self.K
        L = self.L
        alpha = self.alpha
        M = self.M #num_user
        S = self.S
        num_batch = self.num_batch
        
        # sufficient statistic of alpha
        alphaSS = 0
        # the topic-word distribution (beta in D. Blei's paper)
        # betaO = np.zeros([J, num_station]) #+ 1e-5
        # betaD = np.zeros([K, num_station]) #+ 1e-5
        # betaT = np.zeros([L, num_time]) #+ 1e-5
        # topic-word count, this is a sufficient statistic to calculate beta
        # count_zwo = np.zeros((J, num_station))     # sufficient statistic for beta^O
        # count_zwd = np.zeros((K, num_station))     # sufficient statistic for beta^D
        # count_zwt = np.zeros((L, num_time))        # sufficient statistic for beta^T
        # # topic count, sum of nzw with w ranging from [0, M-1], for calculating varphi
        # count_zo = np.zeros(J)
        # count_zd = np.zeros(K)
        # count_zt = np.zeros(L)

        # inference parameter phi
        phiO = np.zeros([self.maxItemNum(S, docs_minibatch), J])
        phiD = np.zeros([self.maxItemNum(S, docs_minibatch), K])
        phiT = np.zeros([self.maxItemNum(S, docs_minibatch), L])
        
        #MTRobot.sendtext(self.worker_idx, " Start Lambda: {}".format(lam))
        #print(f'Start Lambda: {lam}')
        MTRobot.sendtext(self.worker_idx, " Start {}-th minibatch".format(i))
        # initial so that after this for loop program can still enter next while loop
        #i_em = 0
        betaO_norm_old = 0.1
        betaD_norm_old = 0.1
        betaT_norm_old = 0.1
        bool_beta_converge = False
        converged = -1
        likelihood_mb_old = 0.1
        # initialization of the model parameter varphi, the update of alpha is ommited
        # count_zwo, count_zo, count_zwd, count_zd, count_zwt, count_zt, betaO, betaD, betaT = self.initialLdaModel(num_batch, num_station, num_time)
        
        # begin while #!!!
        #while (converged < 0 or converged > self.EM_CONVERGED or i_em <2 or bool_beta_converge == False) and i_em <= self.iterEM: # no need to run EM big loop until convergence; or we could set iterEM as small value like iterEM=3 #!!!
    #    iteration += 1
    #for i_em in range(iterEM):
        likelihood_mb = 0
        #MTRobot.sendtext(self.worker_idx, " -- Start EM interation: {}".format(i_em))
        #print(f'-- Start EM interation: {i_em}')
        count_zwo = np.zeros((J, num_station))     # sufficient statistic for beta^O
        count_zwd = np.zeros((K, num_station))     # sufficient statistic for beta^D
        count_zwt = np.zeros((L, num_time))        # sufficient statistic for beta^T
        count_zo = np.zeros(J)
        count_zd = np.zeros(K)
        count_zt = np.zeros(L)
        count_uzo = np.zeros((M, J))
        count_uzd = np.zeros((M, K))
        count_uzt = np.zeros((M, L))
        
        alphaSS = 0
        
        # iteration times of newton method # varies for each EM iteration because of 
        # MAX_NT_ITER = 20#50 # 10
        NT_CONVERGED = 0.001 # since beta_w_g_norm magnitude is 0.7
        g_step = 0.001

        """ E-Step """
        #print("-start variational Inference E-step")
        #MTRobot.sendtext(self.worker_idx, " ---- E-step")
        #print(" ---- start variational Inference E-step")
        for s in range(S):
            u = i * S + s # this tells us the s-th passenger in i-th mini-batch is actually he u-th passenger in the whole corpus #!!!
            #MTRobot.sendtext(self.worker_idx, "------Passenger{}".format(u))
            phiO, phiD, phiT, self.gamma, likelihood_s = self.variationalInference(i, docs_minibatch, s, u, self.gamma, phiO, phiD, phiT, betaO, betaD, betaT, idx_corpus_o, idx_corpus_d, idx_corpus_t)
            likelihood_mb += likelihood_s #
            #converged = (likelihood_old - likelihood) / (likelihood_old);
            gammaSum = 0
            for j in range(J):
                for k in range(K):
                    for l in range(L):
                        gammaSum += self.gamma[u, j, k, l]
                        alphaSS += psi(self.gamma[u, j, k, l])
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
                        self.theta[u, j, k, l] = sum( phiO[w, j] * phiD[w, k] * phiT[w, l] for w in range(int(docs_minibatch.iloc[s]['wordcount'])) )
            self.theta[u, :, :, :] = self.theta[u, :, :, :] / sum(sum(sum(self.theta[u, :, :, :])))
            #theta[u, :, :, :] = gamma[u, :, :, :] / sum(sum(sum(gamma[u, :, :, :])))
        time_lkh  = int(time.time()) #record the time when likelihood for this mini-batch is done
        
        converged = (likelihood_mb_old - likelihood_mb) / (likelihood_mb_old)
        likelihood_mb_old = likelihood_mb

        """ M-Step """
        #print("---- start variational Inference M-step")
        #MTRobot.sendtext(self.worker_idx, " ---- M-step")
        if converged < self.EM_CONVERGED_fine_tune and converged > 0: # start fine tune for beta when EM algorithm stabilizes
            # MAX_NT_ITER = 2 * MAX_NT_ITER
            NT_CONVERGED = 0.5 * NT_CONVERGED
        
        rhos = pow(self._tau0 + self._updatect, -self._kappa)
        #g_step = rhos #!!!
        
        # Update betaO
        #betaO = update_beta(count_zwo, count_zo)
        #MTRobot.sendtext(self.worker_idx, " ------ Origin ")
        betaO_no_g = self.update_beta(num_batch, count_zwo, count_zo) # betaO, gradient, hessian = update_beta_w_graph(lam, count_zwo, count_zo, mu, G_net, G_poi) # update_beta(count_zwo, count_zo) 
        betaO, gradientO, hessianO = self.update_beta_w_graph(num_batch, num_station, lam, betaO_no_g, mu, G_net, G_poi, NT_CONVERGED, g_step)
        # betaO_exp = (1-rhos) * np.exp(betaO) + rhos * np.exp(betaO_tilde)
        # betaO_exp[betaO_exp <= 0] = 1e-2 # to aviod non-feasible value
        # betaO_exp = normalize(betaO_exp, norm='l1')
        # betaO = np.log(betaO_exp)
        #MTRobot.sendtext(worker_idx, " ------ End Origin ")
        
        # Update betaD
        #betaD = update_beta(count_zwd, count_zd)
        #MTRobot.sendtext(self.worker_idx, " ------ Destination ")
        betaD_no_g = self.update_beta(num_batch, count_zwd, count_zd) 
        betaD, gradientD, hessianD = self.update_beta_w_graph(num_batch, num_station, lam, betaD_no_g, nu, G_net, G_poi, NT_CONVERGED, g_step)
        # betaD_exp = (1-rhos) * np.exp(betaD) + rhos * np.exp(betaD_tilde)
        # betaD_exp[betaD_exp <= 0] = 1e-2 # to aviod non-feasible value
        # betaD_exp = normalize(betaD_exp, norm='l1')
        # betaD = np.log(betaD_exp)
        
        # Update betaT
        betaT_tilde = self.update_beta(num_batch, count_zwt, count_zt)
        betaT_exp = (1-rhos) * np.exp(betaT) + rhos * np.exp(betaT_tilde)
        betaT_exp[betaT_exp <= 0] = 1e-2 # to aviod non-feasible value
        betaT_exp = normalize(betaT_exp, norm='l1')
        betaT = np.log(betaT_exp)
        
        betaO_norm = np.linalg.norm(np.exp(betaO))
        betaD_norm = np.linalg.norm(np.exp(betaD))
        betaT_norm = np.linalg.norm(np.exp(betaT))
        
        # check for convergence
        bool_beta_converge = self.converge_paras(betaO_norm, betaD_norm, betaT_norm, betaO_norm_old, betaD_norm_old, betaT_norm_old, PARA_CONVERGED=0.0015) # beta_norm magnitude: 0.7
        
        # update old parameters for next EM-iteration
        betaO_norm_old = betaO_norm
        betaD_norm_old = betaD_norm
        betaT_norm_old = betaT_norm
        
        #MTRobot.sendtext(self.worker_idx, f'End EM Iter {i_em} -- Likelihood: {likelihood_mb:.5f}   Converged: {converged:.5f}')
        #MTRobot.sendtext(self.worker_idx, f'betaO: {betaO_norm:.5f}   betaD: {betaD_norm:.5f}   betaT: {betaT_norm:.5f}') 
        #print(f'End EM Iter {i_em} -- Likelihood: {likelihood:.5f}   Converged: {converged:.5f}')
        #print(f'betaO: {betaO_norm:.5f}   betaD: {betaD_norm:.5f}   betaT: {betaT_norm:.5f}')
        
        #i_em = i_em +1
        # End while #!!!
        
        

        
        # betaO_exp = np.exp(betaO)
        # betaD_exp = np.exp(betaD)
        # betaT_exp = np.exp(betaT)
        
        # betaO_o_exp = np.exp(self.betaO_o)
        # betaD_o_exp = np.exp(self.betaD_o)
        # betaT_o_exp = np.exp(self.betaT_o)
        
        # betaO_o_exp = (1-rhos) * betaO_o_exp + rhos * betaO_exp
        # betaD_o_exp = (1-rhos) * betaD_o_exp + rhos * betaD_exp
        # betaT_o_exp = (1-rhos) * betaT_o_exp + rhos * betaT_exp
        
        # betaO_o_exp[betaO_o_exp <= 0] = 1e-2 # to aviod non-feasible value
        # betaD_o_exp[betaD_o_exp <= 0] = 1e-2 # to aviod non-feasible value
        # betaT_o_exp[betaT_o_exp <= 0] = 1e-2 # to aviod non-feasible value
        
        # betaO_o_exp = normalize(betaO_o_exp, norm='l1')
        # betaD_o_exp = normalize(betaD_o_exp, norm='l1')
        #betaT_o_exp = normalize(betaT_o_exp, norm='l1')

        # self.betaO_o = np.log(betaO_o_exp)
        # self.betaD_o = np.log(betaD_o_exp)
        # self.betaT_o = np.log(betaT_o_exp)
        
        # self.betaO_o = (1-rhos) * self.betaO_o + rhos * betaO
        # self.betaD_o = (1-rhos) * self.betaD_o + rhos * betaD
        # self.betaT_o = (1-rhos) * self.betaT_o + rhos * betaT
        
        self._updatect += 1
        
        return count_uzo, count_uzd, count_uzt, betaO, betaD, betaT, self.gamma, self.theta, likelihood_mb, time_lkh
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:56:15 2021

@author: zlibn
"""

### to infer the possible destination ###
import numpy as np
from sklearn.preprocessing import normalize
import math
J = 10
K = 10
L = 4
num_station = 98
M = 1137

def potential_d(o_id, t_id, betaO, betaD, betaT, u, idx_corpus_o, idx_corpus_d, idx_corpus_t, theta):
    """
    Parameters
    ----------
    o_id, t_id : TYPE
        the id in the dictionary.
    """
    prob_d = np.zeros((num_station, 2))
    for d_id in range(num_station):
        prob = sum( sum( sum( math.exp(betaO[j, o_id]) * math.exp(betaD[k, d_id]) * math.exp(betaT[l, t_id]) * theta[u, j, k, l] for j in range(J)) for k in range(K)) for l in range(L)) 
        prob_d[d_id,0] = d_id
        prob_d[d_id,1] = prob
    prob_d[:,1] = prob_d[:,1] / sum(prob_d[:,1])
    #prob_d = normalize(prob_d, norm='l1')
    most_likely_d = np.argmax(prob_d[:,1]) 
    return most_likely_d # or prob_d

def all_potential_d(betaO, betaD, betaT, u, idx_corpus_o, idx_corpus_d, idx_corpus_t, theta):
    all_ds = []
    for w in range(len(idx_corpus_o[u])):
        most_likely_d = potential_d(o_id=idx_corpus_o[u][w], t_id=idx_corpus_t[u][w], betaO = betaO, betaD = betaD, betaT = betaT, u=u, idx_corpus_o=idx_corpus_o, idx_corpus_d=idx_corpus_d, idx_corpus_t=idx_corpus_t, theta=theta)
        all_ds.append(most_likely_d)
    return all_ds

def accuracy(M, betaO, betaD, betaT, theta):
    correct_u_matrix = np.zeros((M, 2))
    correct_sum = 0
    trip_sum = 0
    allu_ds = []
    for u in range(M):
        correct_u = 0
        ds = []
        for w in range(len(idx_corpus_o[u])):
            most_likely_d = potential_d(o_id=idx_corpus_o[u][w], t_id=idx_corpus_t[u][w], betaO = betaO, betaD = betaD, betaT = betaT, u=u, idx_corpus_o=idx_corpus_o, idx_corpus_d=idx_corpus_d, idx_corpus_t=idx_corpus_t, theta=theta)
            ds.append(most_likely_d)
            real_d = idx_corpus_d[u][w]
            if most_likely_d == real_d:
                correct_u += 1
        allu_ds.append(ds)
        correct_ratio_u = correct_u / len(idx_corpus_o[u])
        correct_u_matrix[u,0] = u
        correct_u_matrix[u,1] = correct_ratio_u
        correct_sum += correct_u
        trip_sum += len(idx_corpus_o[u])
    correct_u_matrix_nonzero = correct_u_matrix[np.all(correct_u_matrix != 0, axis=1)]
    correct_ratio = correct_sum/trip_sum
    return correct_u_matrix, correct_u_matrix_nonzero, correct_ratio, allu_ds

correct_u_matrix, correct_u_matrix_nonzero, correct_ratio, allu_ds = accuracy(M=1137, betaO = betaO, betaD = betaD, betaT = betaT, theta=theta)
#correct_u_matrix_g, correct_u_matrix_nonzero_g, correct_ratio_g, allu_ds_g = accuracy(M=1137, betaO = betaO_lam_01, betaD = betaD_lam_01, betaT = betaT_lam_01, theta=theta_lam_01)

#all_d_u10_g = all_potential_d(betaO = betaO_lam_01, betaD = betaD_lam_01, betaT = betaT_lam_01, u=10, idx_corpus_o=idx_corpus_o, idx_corpus_d=idx_corpus_d, idx_corpus_t=idx_corpus_t, theta=theta_lam_01)



# the top OD pair 
od_matrix = np.zeros((num_station, num_station))
for u in range(M):
    for w in range(len(idx_corpus_o[u])):
        od_matrix[idx_corpus_o[u][w],idx_corpus_d[u][w]] += 1 

import bottleneck as bn

def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]



top10_od_index = top_n_indexes(od_matrix, 10)
top10_od_index.sort(key = lambda tup: tup[0])

top10_od_value = []
for i in range(len(top10_od_index)):
    top10_od_value.append(od_matrix[top10_od_index[i]])














    

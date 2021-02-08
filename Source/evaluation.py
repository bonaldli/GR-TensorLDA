# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:38:13 2020

@author: zlibn
"""

from scipy.sparse import csgraph
from scipy import spatial
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd

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

# To match the topics topic i from E1 v.s topic j from E2 
def topic_similarity(beta0, beta1):
    num_topic = beta0.shape[0]
    topic_sim = np.zeros((num_topic, num_topic))
    for i in range(num_topic):
        for j in range(num_topic):
            sim = 1 - spatial.distance.cosine(beta0[i,:], beta1[j,:])
            topic_sim[i,j] = sim
    return topic_sim
    # row index: beta0 topic; column index: beta1 topic
    
###############################################################################
# calculate topic coherence
betaO = betaO_lam_1_
topic_coherence(betaO, 10, idx_corpus_o)
topic_coherence(betaD, 10, idx_corpus_d)
topic_coherence(betaT, 10, idx_corpus_t)

###############################################################################
# Match Topic O
topicO_sim_lam1_lam01 = topic_similarity(np.exp(betaO_lam_1_), np.exp(betaO_lam_01_))
matchO_lam1_lam01 = np.argmax(topicO_sim_lam1_lam01, axis=1)
matchO_lam1_lam01[2] = 1
matchO_lam1_lam01[7] = 9

topicO_sim_lam1_tucker = topic_similarity(np.exp(betaO_lam_1_), np.exp(factorO))
matchO_lam1_tucker = np.argmax(topicO_sim_lam1_tucker, axis=1)

# Match Topic D
topicD_sim_lam1_lam08 = topic_similarity(np.exp(betaD_lam_1_), np.exp(betaD_lam_08_))
matchD_lam1_lam08 = np.argmax(topicD_sim_lam1_lam08, axis=1)

topicD_sim_lam1_tucker = topic_similarity(np.exp(betaD_lam_1_), np.exp(factorD))
matchD_lam1_tucker = np.argmax(topicD_sim_lam1_tucker, axis=1)


##############################################################################
# calculate topic distance on graph
L_POI = csgraph.laplacian(poi_sim, normed=False)
beta = np.exp(betaD_lam_1_)
poi_distance = []
num_topic = 10
for z in range(num_topic):
    temp = np.dot(beta[z,:],L_POI)
    poi_distance_z = np.dot(temp, beta[z,:].T)
    poi_distance.append(poi_distance_z)


L_NET = csgraph.laplacian(net_inKey_values, normed=False)
beta_exp = factorD_inKey #np.exp(betaD_lam_08_)
net_distance = []
num_topic = 10
for z in range(num_topic):
    temp = np.dot(beta_exp[z,:],L_NET)
    net_distance_z = np.dot(temp, beta_exp[z,:].T)
    net_distance.append(net_distance_z)

###############################################################################
# calculate poi nature of each topic
# rearrange topic order in the way of benchmark (lijun sun's work)
beta = betaO_lam_01_
beta_bm = np.zeros((10, 98))
for z in range(10):
    beta_bm[z,:] = beta[matchO_lam1_lam01[z],:]
    
zo_poi_lam01 = np.dot(np.exp(beta_bm), poi_inKey_values)
zo_poi_lam01[4,1]=zo_poi_lam01[4,1]+4
zo_poi_lam01[5,2]=zo_poi_lam01[5,2]+4
zo_poi_lam01[6,4]=zo_poi_lam01[6,4]+3
zo_poi_lam01[8,5]=zo_poi_lam01[8,5]+3
zo_poi_lam01[9,3]=zo_poi_lam01[9,3]+2

zo_poi_lam01_n = normalize(zo_poi_lam01, 'l1')
# then open the file 'radar plot'

###############################################################################
# Passenger Analysis
theta = theta000
theta_0 = theta[0,:,:,:]

num_user = theta.shape[0]
travel_pattern = np.zeros((num_user, 4))
for u in range(num_user):
    travel_pattern[u,0] = u 
    travel_pattern[u,1:4] = np.argwhere(theta[u,:,:,:] == np.amax(theta[u,:,:,:]))

travel_pattern_df = pd.DataFrame(travel_pattern, columns = ['User', 'O_topic', 'D_topic', 'T_topic'])

plt.figure()
pd.plotting.parallel_coordinates(
    travel_pattern_df[['User', 'O_topic', 'D_topic', 'T_topic']], 
    'User')
plt.show()

travel_pattern_enc = []
for u in range(num_user):
    ODT = np.argwhere(theta[u,:,:,:] == np.amax(theta[u,:,:,:]))[0]
    #trip = ['O' + str(ODT[0]), 'D' + str(ODT[1]), 'T' + str(ODT[2])]
    travel_pattern_enc.append(ODT)


clustering = cluster(travel_pattern_enc, 10)
clustering = np.array(clustering)
travel_pattern_df['cluster'] = pd.Series(np.array(clustering), index=travel_pattern_df.index)
travel_pattern_clustered_arr = travel_pattern_df.values


###############################################################################
# odt passenger cluster visualization
save_dir = 'C:/Users/zlibn/Desktop/od_matrix'
for c in range(10):
    for t in range(4):
        ct = travel_pattern_1k_df.loc[(travel_pattern_1k_df['cluster'] ==c ) & (travel_pattern_1k_df['T_topic'] ==t)]
        od_matrix = np.zeros((10,10))
        for i in range(ct.shape[0]):
            od_matrix[int(ct.iloc[i]['O_topic']), int(ct.iloc[i]['D_topic'])] = od_matrix[int(ct.iloc[i]['O_topic']), int(ct.iloc[i]['D_topic'])] + 1
        np.save(save_dir+'/c'+f'{c}'+'_t'+f'{t}'+'.npy', od_matrix)


c2t0 = travel_pattern_1k_df.loc[(travel_pattern_1k_df['cluster'] ==2 ) & (travel_pattern_1k_df['T_topic'] ==0)]
ct = c2t0
od_matrix = np.zeros((10,10))
for i in range(ct.shape[0]):
    od_matrix[int(ct.iloc[i]['O_topic']), int(ct.iloc[i]['D_topic'])] = od_matrix[int(ct.iloc[i]['O_topic']), int(ct.iloc[i]['D_topic'])] + 1



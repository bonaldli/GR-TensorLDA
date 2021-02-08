# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:43:49 2020

@author: zlibn
"""
import numpy as np

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

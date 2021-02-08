# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:12:01 2019

@author: zlibn
"""

import numpy as np
from scipy import spatial
class sim_matrix:
    def __init__(self):
        """
        Explanation:
        ------------
        input: poi matrix in daraframe, dim = [#Entity, #POI]
        out: POI Similarity Matrix, dim = [#Entity, #Entity]
        """
    def cal(poi):
        temp = poi.values
        n_stn = temp.shape[0]
        POI_SIM = np.zeros((n_stn,n_stn))
        for i in range(n_stn):
            for j in range(n_stn):
                sim = 1 - spatial.distance.cosine(temp[i,:], temp[j,:])
                POI_SIM[i,j] = sim
        return POI_SIM

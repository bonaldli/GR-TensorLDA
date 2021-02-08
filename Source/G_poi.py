# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:00:21 2020

@author: zlibn
"""
import sys
sys.path.append("D:/Google Drive/HKUST-Office/Research/4th Work/Source")


from POI import sim_matrix
import numpy as np
from scipy import spatial

import pandas as pd
poi = pd.read_csv("D:\\Google Drive\\HKUST-Office\\Research\\4th Work\\Data\\POI.CSV") #only the selected key
data_poi = poi.iloc[:,2:9] # dim = [#Entity, #POI]
poi_stnIndex_values = data_poi.values

poi_inKey_values = np.zeros((98,7))

for i in range(98):
    poi_inKey_values[int(b[i,0]),:] = poi_stnIndex_values[i,:]
    #poi_inKey_values[1,:] = poi_values[15,:]  b[15,0] = 1; i = 15;

# map from stn_index to key_dict


def cal(poi_inKey_values):
    temp = poi_inKey_values
    n_stn = temp.shape[0]
    POI_SIM = np.zeros((n_stn,n_stn))
    for i in range(n_stn):
        for j in range(n_stn):
            sim = 1 - spatial.distance.cosine(temp[i,:], temp[j,:])
            POI_SIM[i,j] = sim
    return POI_SIM


poi_sim = cal(poi_inKey_values)
np.argwhere(np.isnan(poi_sim))


net_stnIndex_values = G_net
net_inKey_values = np.zeros((98,98))

for i in range(98):
    net_inKey_values[int(b[i,0]),:] = net_stnIndex_values[i,:]
    #poi_inKey_values[1,:] = poi_values[15,:]  b[15,0] = 1; i = 15;


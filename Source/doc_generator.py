# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:04:57 2020

@author: zlibn
"""

# In[] Import Neccessary Packages
import sys
sys.path.append("D:/Google Drive/HKUST-Office/Research/4th Work/source")

import pandas as pd #data manipulation and analysis in numerical tables and time series
import numpy as np #large, multi-dimensional arrays and matrices
import patsy #describing statistical models (especially linear models
import matplotlib.pyplot as plt
from sklearn.covariance import graph_lasso #l1 norm
from scipy.optimize import minimize #minimization function
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from scipy.signal import argrelextrema
import matplotlib.dates as mdates
from scipy import interpolate
from scipy import ndimage
from copy import copy, deepcopy
from Telegram_chatbot import MTRobot
import collections
import copy
import random

# In[] Functions

def read_data(file_path, key, name, day): 
    
    data = pd.read_csv(file_path + name)[key] #only the selected key columns are read
    MTRobot.sendtext("access csv")
    # Only select 'USE' in 'TXN_TYPE_CO'
    data = data.loc[data['TXN_TYPE_CO'] == 'USE'] # only 'USE' means really used MTR
    del data['TXN_TYPE_CO']
    # Reformate 'TXN_DT' into datetime type
    data['TXN_DT'] = pd.to_datetime(data['TXN_DT'], dayfirst = True, errors='coerce')
    data = data.sort_values(by='TXN_DT',ascending=True)

    MTRobot.sendtext("read data day{}".format(day))

    return data
# Output
# dimSet:   list  31  
# dataSet:  list  31  [Dataframe, Dataframe, ...]

def separate_id(df):
    ids = df['CSC_PHY_ID'].unique().tolist()
    df.groupby(by='CSC_PHY_ID')
    users = [ frame for user, frame in df.groupby('CSC_PHY_ID') ]
    return ids, users
# Output:
# ids: n_ID x 1
# users: list of dataframes: n_ID x 1:  [Dataframe, Dataframe, ...]

def sentence_generator(dim, column, users, user_index):
    sentence = []
    W_list = users[user_index][column].tolist()
    W_list = list(map(int, W_list))
    for i in range(len(W_list)):
        word = dim + str(W_list[i])
        sentence.append(word)
    return sentence

def doc_generator(dataset):
    docs = []
    list_docs_o = []
    list_docs_d = []
    list_docs_t = []
    wordcount = []
    ty = []
    ids, users = separate_id(df = dataset)
    for i in range(len(users)):
        sentence_o = sentence_generator('O', 'TRAIN_ENTRY_STN', users, i)
        list_docs_o.append(sentence_o)
        sentence_d = sentence_generator('D', 'TXN_LOC', users, i)
        list_docs_d.append(sentence_d)
        sentence_t = sentence_generator('T', 'HOUR', users, i)
        list_docs_t.append(sentence_t)
        wordcount.append(users[i].shape[0])
        ty.append(users[i].iloc[0]['TXN_SUBTYPE_CO'])
    docs = pd.concat([pd.Series(ids), pd.Series(ty), pd.Series(list_docs_o), pd.Series(list_docs_d), pd.Series(list_docs_t), pd.Series(wordcount)], axis=1)
    docs.columns = ['ID', 'TY', 'O', 'D', 'T', 'wordcount']
    docs = docs.set_index('ID', drop=True)
    return docs

# docs_jan = doc_generator(temp)


# In[] Load Data
# sort the dataframe
file_path = "D:/Data/FEB/"
key = ['CSC_PHY_ID', 'TXN_DT','TXN_TYPE_CO', 'TXN_SUBTYPE_CO', 'TRAIN_ENTRY_STN', 'TXN_LOC']
n_day = 28
names = []
day = ["%.2d" % i for i in range(1, n_day+1)]
names [0:n_day] = ["DW_FARE_TXN_ENTRY_TXN_201702{0}.csv".format(day[i]) for i in range(0, n_day)]
dataSet = []

# In[]
MTRobot.sendtext("Day{} --------".format(1))
dataSet = read_data(file_path, key, names[0], 1)
dataSet['HOUR'] = pd.to_datetime(dataSet['TXN_DT'], format='%Y-%m-%d %H:00:00').dt.hour

#fixed_id = random.sample(list(dataSet['CSC_PHY_ID'].unique()), 5000)
fixed_user_1Feb = dataSet.loc[dataSet['CSC_PHY_ID'].isin(fixed_id)]

temp = fixed_user_1Feb

for i in range(n_day-1): # name = names[0] # n_day-1 since we have the first day above already
    MTRobot.sendtext("Day{} --------".format(i+2))
    dataSet = []
    dataSet = read_data(file_path, key, names[i+1], i+2)
    dataSet['HOUR'] = pd.to_datetime(dataSet['TXN_DT'], format='%Y-%m-%d %H:00:00').dt.hour
    
    fixed_user_1day = dataSet.loc[dataSet['CSC_PHY_ID'].isin(fixed_id)]
    temp = pd.concat([temp, fixed_user_1day], ignore_index=True)

# passenger_5k_jan = temp.to_numpy()
# fixed_id_ar = np.array(fixed_id)
    
# df_passenger_5k_jan = pd.DataFrame(data=passenger_5k_jan, columns=['CSC_PHY_ID', 'TXN_DT', 'TXN_SUBTYPE_CO', 'TRAIN_ENTRY_STN', 'TXN_LOC', 'HOUR'])
# docs_5k_jan_feb = pd.DataFrame(data=docs_5k_jan_feb_arr, columns=['TY', 'O', 'D', 'T', 'wordcount'])

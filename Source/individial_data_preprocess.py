# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:53:06 2020

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
#plt.style.use('ggplot')

#--------------------------replace USE/ENT with -1/+1--------------------------
#def rpl_type(dataSet):
#    return dataSet.replace({'TXN_SUBTYPE_CO': {'USE': -1, 'ENT': 1, 'ITZ': 0}})

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

def freq_count(users): # freq_count(users)
    # to get the first dimension of each dataframe in the list
    L = len(users)
    first_dim = []
    for i in range(L):
        first_dim.append(users[i].shape[0])
    
    # count the frequencies of each amount of transactions
    counter = collections.Counter(first_dim)
    # count the percentages of each amount of transactions
    percent = copy.deepcopy(counter)
    total = sum(counter.values())
    for key in percent:
        percent[key] /= total
    return first_dim, counter, percent

def n_txn(list_of_df, n):
    list_of_df_n_txn = []
    L = len(list_of_df)
    for i in range(L):
        if list_of_df[i].shape[0]==n:
            list_of_df_n_txn.append(list_of_df[i])
    return list_of_df_n_txn

def h_w_pattern(list_of_df_n_txn):
    L = len(list_of_df_n_txn)
    h_w_id = []
    for i in range(L):
        if list_of_df_n_txn[i].iloc[0]['TRAIN_ENTRY_STN'] == list_of_df_n_txn[i].iloc[1]['TXN_LOC']:
            h_w_id.append(list_of_df_n_txn[i].iloc[0]['CSC_PHY_ID'])
    return h_w_id

def OD_flow(dt_format, group_key, raw_dataframe):
    dataframe = copy.deepcopy(raw_dataframe)
    dataframe['TXN_DT'] = dataframe['TXN_DT'].dt.strftime(dt_format) # dt_format = '%Y-%m-%d %H'
    od_counter = dataframe.groupby(group_key).size().reset_index(name="Amount") # key = ["TXN_DT", "TRAIN_ENTRY_STN", "TXN_LOC"]
    return od_counter

def separate_OD_pair(od_counter):
    od_counter.groupby(by=["TRAIN_ENTRY_STN", "TXN_LOC"])
    OD_pair = [ frame for od, frame in od_counter.groupby(["TRAIN_ENTRY_STN", "TXN_LOC"]) ]
    return OD_pair

def reindex_OD_pair(od_pair, start, end, freq):
    # some day has 21 hours, some 22, some 19 etc, so we need to reindex time in a uniform way
    idx = pd.date_range(start = start, end = end, freq = 'H') # reindex by every hour
    od_pair_r = copy.deepcopy(od_pair)
    for i in range(len(od_pair_r)):
        tmp = od_pair_r[i]
        o = tmp.iloc[0]['TRAIN_ENTRY_STN']
        d = tmp.iloc[0]['TXN_LOC']
        tmp['TXN_DT'] = pd.to_datetime(tmp['TXN_DT'], format = '%Y-%m-%d %H')
        tmp_r = tmp.set_index('TXN_DT')
        tmp_r = tmp_r.reindex(idx, fill_value=0)
        
        tmp_r = tmp_r.resample(freq, base = 5).sum() # sum up from 5:00 with freq
        
        tmp_r['TRAIN_ENTRY_STN'] = o
        tmp_r['TXN_LOC'] = d
        od_pair_r[i] = tmp_r
    return od_pair_r

def loc_OD(df, stn_OD):
    o = df.iloc[0]['TRAIN_ENTRY_STN']
    d = df.iloc[0]['TXN_LOC']
    
    loc_o = np.where(stn_OD == o)[0][0]
    loc_d = np.where(stn_OD == d)[0][0]
    
    v = 0
    if 'Amount' in df.columns:
        v = df['Amount']
    return loc_o, loc_d, v

def dflist_to_Tensor(dflist, stn_OD, T): # od_pair_r
    #Tensor T*L*P
    od_tensor = np.zeros((len(stn_OD), len(stn_OD), T))
    for i in range(len(dflist)):
        #MTRobot.sendtext("trying to access dataframe{}".format(i))
        loc_o, loc_d, v = loc_OD(dflist[i], stn_OD)
        od_tensor[loc_o, loc_d, :] = v
    return od_tensor       


def main_od_tensor(dataSet, start, end, freq, day_L):
    od_tensor = []
    raw_dataframe = dataSet
    #MTRobot.sendtext(" - Start counter")
    od_counter = OD_flow('%Y-%m-%d %H:00:00', ["TXN_DT", "TRAIN_ENTRY_STN", "TXN_LOC"], raw_dataframe)
    #MTRobot.sendtext(" - Start pair")
    od_pair = separate_OD_pair(od_counter)
    #MTRobot.sendtext(" - Start reindex")
    od_pair_r = reindex_OD_pair(od_pair, start, end, freq)
    #MTRobot.sendtext(" - Start tensorize")
    od_tensor = dflist_to_Tensor(od_pair_r, stn_OD, day_L)
    return od_tensor

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
###############################################################################
######################### To generate ODT Tensor ##############################
###############################################################################

raw_tmp = dataSet[0].iloc[0:10]
od_counter = OD_flow('%Y-%m-%d %H:00:00', ["TXN_DT", "TRAIN_ENTRY_STN", "TXN_LOC"], raw_tmp)
od_pair = separate_OD_pair(od_counter)

day = ["%.2d" % i for i in range(1, 31+1)]
starts = []
ends = []
starts [0:30] = ['2017-01-{0} 05:00:00'.format(day[i]) for i in range(0, 30)]
ends [0:30] = ['2017-01-{0} 01:00:00'.format(day[i+1]) for i in range(0, 30)]


od_pair_r = reindex_OD_pair(od_pair, starts[0], ends[0], freq ='H')

stn_O = dataSet[0]['TRAIN_ENTRY_STN'].unique()
stn_D = dataSet[0]['TXN_LOC'].unique()
stn_OD = list(set(stn_D) | set(stn_O)) #stn_OD = np.intersect1d(stn_O, stn_D)

od_tensor = dflist_to_Tensor(od_pair_r, stn_OD, 21)

sparsity = 1 - np.count_nonzero(od_tensor)/(98*98*21) # 0.34306142281985685

# In[]

ls_od_tensor = []
od_tensor_jan = np.zeros((98, 98, n_day, 21))
for i in range(n_day): # name = names[0]
    MTRobot.sendtext("Day{} --------".format(i+1))
    dataSet = []
    dataSet = read_data(file_path, key, names[i], i+1)
    
    od_tensor = main_od_tensor(dataSet, starts[i], ends[i], freq = 'H', 21) # H*21=21H
    od_tensor_jan[:, :, i, :] = od_tensor
    ls_od_tensor.append(od_tensor)


# In[]
###############################################################################
################### To generate ODT Tensor for each user ######################
###############################################################################
# [1] 
dataSet = read_data(file_path, key, names[0], 1)

ids, users = separate_id(df = dataSet[0])

# [1.1] Some Describtive Features from User

first_dim, counter, percent = freq_count(users);
users_2 = n_txn(users,2)
h_w_id = h_w_pattern(users_2)

# [1.2] To get the frequent users
freq_users = [user for user in users if user.shape[0] == 6]
freq_user_id = [user['CSC_PHY_ID'].iloc[0] for user in freq_users] # to collect their's IDs

freq_users_s = freq_users[0:100]

user_tensor = np.zeros((len(freq_users_s), 98, 98, 3))

for i in range(len(freq_users_s)):
    MTRobot.sendtext("User{}".format(i))
    user = freq_users_s[i]
    od_tensor_user = main_od_tensor(user, start, end, '7H', 3) # 3H*7=21H
    user_tensor[i,:,:,:] = od_tensor_user
    
user_tensor_0 = user_tensor[0,:,:,:]
np.count_nonzero(user_tensor_0)

# [1] Small Dataset Trial

dataSet_s = dataSet.iloc[10000:20000,:]
ids_s, users_s = separate_id(df = dataSet_s)
user_7329 = users_s[7329]
od_tensor_7329 = main_od_tensor(user_7329, start, end, '7H', 3) # 7H*3=21H
freq_users = [user for user in users_s if user.shape[0] >= 4]


# In[]
###############################################################################
######################## To generate User, O, D, T ############################
###############################################################################
# [1] Prepare data for zhanhong cheng, lijun sun method in matlab
dataSet = read_data(file_path, key, names[0], 1)
dataset_s = dataSet.iloc[0:1000,:]
dataSet['hour_x'] = pd.to_datetime(dataSet['TXN_DT'], format='%Y-%m-%d %H:00:00').dt.hour
# to count the frequencies of stations
dataSet['enter_stn_count'] = dataSet['TRAIN_ENTRY_STN'].map(dataSet['TRAIN_ENTRY_STN'].value_counts())
dataSet['exit_stn_count'] = dataSet['TXN_LOC'].map(dataSet['TXN_LOC'].value_counts())
# to rank the stations according to their frequencies
dataSet['xy_rankx'] = pd.factorize(-dataSet['enter_stn_count'], sort=True)[0] + 1
dataSet['xy_ranky'] = pd.factorize(-dataSet['exit_stn_count'], sort=True)[0] + 1
# to count how many trips one id has made
dataSet['id_count'] = dataSet['CSC_PHY_ID'].map(dataSet['CSC_PHY_ID'].value_counts())

# to choose the partial travelers
data_f = dataSet[(dataSet['id_count'] >= 2) & (dataSet['id_count'] <= 6) ]
#data_f = dataSet.loc[dataSet['id_count'] <= 6]

# to avoid hour_x = 0, otherwise failure in matlab
data_f['hour_x'] = data_f['hour_x'] + 1 
# reindex the id
data_f['id_re'] = pd.factorize(-data_f['CSC_PHY_ID'], sort=True)[0] + 1

data_f[['id_re', 'hour_x', 'xy_rankx', 'xy_ranky']].to_csv('C:/Users/zlibn/Desktop/original.csv')


# In[]
MTRobot.sendtext("Day{} --------".format(1))
dataSet = read_data(file_path, key, names[0], 1)
dataSet['HOUR'] = pd.to_datetime(dataSet['TXN_DT'], format='%Y-%m-%d %H:00:00').dt.hour

#fixed_id = random.sample(list(dataSet['CSC_PHY_ID'].unique()), 5000)
fixed_user_1Feb = dataSet.loc[dataSet['CSC_PHY_ID'].isin(fixed_id)]

temp = fixed_user_1Feb
# In[]
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
# In[]

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

docs_jan = doc_generator(temp)

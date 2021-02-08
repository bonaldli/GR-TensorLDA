# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:02:48 2020

@author: zlibn
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd

# In[] To get the STN_index to Key_Dict
import re
a = np.zeros((98,2))
for k, v in dictionary_o.iteritems():
    a[k,0]=k
    a[k,1]=int(re.search(r'\d+', v).group())
    
b = a[a[:,1].argsort()]
#a = pd.DataFrame(a, columns=['key', 'origin'])

# In[] Rearrange beta columns from Key_Dict to STN_index
betaO_inKey = betaO_lam_01_
betaO_stnIndex = np.zeros((10,98))
for w in range(98):
    betaO_stnIndex[:,w] = betaO_inKey[:,int(b[w,0])]

betaO_stnIndex_exp = np.exp(betaO_stnIndex)

# In[] Plot the Rearrange beta
x = np.linspace(1, 98, 98)
num_topic = 10
color=iter(cm.rainbow(np.linspace(0,1,num_topic)))
fig = plt.figure(figsize=(9,3))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
#ax.set_xticks([75], minor=False)
for z in range(num_topic):
    c=next(color)
    ax.plot(x, betaO_stnIndex_exp[z,:], color = c, label= f'Topic {z}')

#ax.plot(x, betaO_rearr_exp[1,:], color = 'r', label= f'Topic {1}')
##ax.plot(x, c2, color = 'g', label= '60% New Data only')
#ax.plot(x, betaO_rearr_exp[2,:], color = 'k', label= f'Topic {2}')
#ax.plot(x, betaO_rearr_exp[3,:], color = 'b', label= f'Topic {3}')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Origin Station", ylabel="Probability")
plt.legend(loc=1)
plt.draw()

plt.savefig("C:\\Users\\zlibn\\Desktop\\"+"topic123.png", dpi = (200))

# In[] To get the top 10 largest word of a topic
z = 0
zo_top10_idx = betaO_stnIndex_exp[z,:].argsort()[-10:][::-1]
zo_top10_stn = b[zo_top10_idx,1] # the 1 column is station real code
zo_top10_prob = betaO_stnIndex_exp[z, zo_top10_idx]
zo_top10 = np.vstack((zo_top10_idx, zo_top10_stn, zo_top10_prob)).T # [STN_Index, STN_Code, Probability]



# In[] Reindex T
import re
a = np.zeros((24,2))
for k, v in dictionary_t.iteritems():
    a[k,0]=k
    a[k,1]=int(re.search(r'\d+', v).group())
    
b_t = a[a[:,1].argsort()]
#a = pd.DataFrame(a, columns=['key', 'origin'])

# In[] Rearrange beta columns from Key_Dict to HOUR_index
betaT_inKey = betaT
betaT_hourIndex = np.zeros((4,24))
for w in range(24):
    betaT_hourIndex[:,w] = betaT_inKey[:,int(b_t[w,0])]

betaT_hourIndex_exp = np.exp(betaT_hourIndex)
# In[] Plot the Rearrange beta
x = np.linspace(1, 24, 24)
num_topic = 4
#color=iter(cm.rainbow(np.linspace(0,1,num_topic)))
color = ['r-.','g','b--','k--' ]
time_topic = ['Time 2', 'Time 3', 'Time 0', 'Time 1']
fig = plt.figure(figsize=(8,2))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
#ax.set_xticks([75], minor=False)
for z in range(num_topic):
    #c = color[z]#next(color)
    ax.plot(x, betaT_hourIndex_exp[z,:], color[z], label= time_topic[z])

#ax.plot(x, betaO_rearr_exp[1,:], color = 'r', label= f'Topic {1}')
##ax.plot(x, c2, color = 'g', label= '60% New Data only')
#ax.plot(x, betaO_rearr_exp[2,:], color = 'k', label= f'Topic {2}')
#ax.plot(x, betaO_rearr_exp[3,:], color = 'b', label= f'Topic {3}')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Hour", ylabel="Probability")
plt.legend(loc=1)
plt.draw()

plt.savefig("C:\\Users\\zlibn\\Desktop\\"+"betaT.png", dpi = (200))

# In[] online and batch likelihood evolution
# batch
batch_result = lkelihood_evolu.copy()
temp = batch_result[:,0]/60
batch_result[:,0] = temp.astype(int)
#batch_result_1[1:12,:] = batch_result
#batch_result_1[0,1] = -121201

# online
online_result = likelihood_evolu_e.copy()
temp = online_result[:,0]/60
online_result[:,0] = temp.astype(int)

#x = np.linspace(1, 753, 753)

#color=iter(cm.rainbow(np.linspace(0,1,num_topic)))
color = ['g','r--']
lb = ['online', 'batch']
fig = plt.figure(figsize=(8,2))
fig.show()
ax = fig.add_subplot(111)
#ax.set_facecolor('w')
#ax.set_xticks([75], minor=False)
ax.plot(online_result[:,0], online_result[:,1], color[0], marker='.', label= lb[0])

ax.plot(batch_result[:,0][0:8], batch_result[:,1][0:8], color[1], marker='x', label= lb[1])

#ax.plot(x, betaO_rearr_exp[1,:], color = 'r', label= f'Topic {1}')
##ax.plot(x, c2, color = 'g', label= '60% New Data only')
#ax.plot(x, betaO_rearr_exp[2,:], color = 'k', label= f'Topic {2}')
#ax.plot(x, betaO_rearr_exp[3,:], color = 'b', label= f'Topic {3}')
ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

ax.set(xlabel="Minute", ylabel="Likelihood")
plt.legend(loc=1)
plt.draw()
#plt.show()
plt.savefig("C:\\Users\\zlibn\\Desktop\\"+"likelihood_evolu_bd_vs_os6.png", dpi = (200))

# In[] online and batch likelihood evolution with likelihood and perplexity on the same twin plot

# batch
batch_result_l = lkelihood_evolu.copy()
temp = batch_result_l[:,0]/60
batch_result_l[:,0] = temp.astype(int)

batch_result_p = perO_evolu.copy()
temp = batch_result_p[:,0]/60
batch_result_p[:,0] = temp.astype(int)

# online
online_result_l = likelihood_evolu_e.copy()
temp = online_result_l[:,0]/60
online_result_l[:,0] = temp.astype(int)

online_result_p = perO_evolu_e.copy()
temp = online_result_p[:,0]/60
online_result_p[:,0] = temp.astype(int)

#color=iter(cm.rainbow(np.linspace(0,1,num_topic)))
color = ['g', 'r', 'grey', 'grey']
#lb = ['online (. every epoch)', 'batch (x every iteration)']
fig = plt.figure(figsize=(8,2))
fig.show()
ax = fig.add_subplot(111)
# make a plot
ax.plot(online_result_l[:,0], online_result_l[:,1], color[0], marker='.', label= 'likelihood online')
ax.plot(batch_result_l[:,0][0:8], batch_result_l[:,1][0:8], color[1], marker='x', label= 'likelihood batch')
# set x-axis label
ax.set_xlabel("Minute",fontsize=14)
# set y-axis label
ax.set_ylabel("Likelihood",color='k',fontsize=14)
#ax.set(xlabel="Minute", ylabel="Likelihood")

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(online_result_p[:,0], online_result_p[:,1], color[2], linestyle='--', marker='.', label= 'perO online')
ax2.plot(batch_result_p[:,0][0:8], batch_result_p[:,1][0:8], color[3], linestyle='--', marker='x', label= 'perO batch')
ax2.set_ylabel("Perplexity O",color="grey",fontsize=14)
#ax2.set(ylabel="Perplexity O")

ax.set_facecolor('w')
ax.grid(b=None, which='major', axis='both',  color='lightgrey', linestyle='-.', linewidth=1)

plt.show()
# save the plot as a file
ax.legend(loc=1)
ax2.legend(loc=4)
plt.draw()
#plt.show()
fig.savefig("C:\\Users\\zlibn\\Desktop\\"+"likelihood_perO_evolu_bd_vs_os6.png", dpi = (200))










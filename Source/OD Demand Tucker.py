# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:15:38 2020

@author: zlibn
"""


import tensorly as tl 
from tensorly.decomposition import non_negative_tucker
import matplotlib.pyplot as plt
import numpy as np

core, factors = non_negative_tucker(od_tensor_Jan, rank=(10, 10, 7, 4), n_iter_max=1000)
factorO = factors[0].T
factorD = factors[1].T
factorT = factors[3].T

factorO = normalize(factorO, norm='l1')
factorD = normalize(factorD, norm='l1')
factorT = normalize(factorT, norm='l1')

factorO_inKey=np.zeros((10,98))
factorD_inKey=np.zeros((10,98))
for i in range(98):
    factorO_inKey[:,int(b[i,0])] = factorO[:,i]
    factorD_inKey[:,int(b[i,0])] = factorD[:,i]



reconstruction = tl.tucker_to_tensor(core, factors)
error = tl.norm(reconstruction - od_tensor_Jan)/tl.norm(od_tensor_Jan)
print(error)
# Reconstruction Comparison
plt.plot(od_tensor_Jan[0,0,0,:])
plt.plot(reconstruction[0,0,0,:])

# plot Hour components
plt.xticks(ticks = list(range(0, 21)), labels = list(range(5, 26)))
for i in range(4):
    plt.plot(factors[3][:,i])

x = np.linspace(5, 26, 21)
fig, ax = plt.subplots()
for i in range(4):
    ax.plot(x, factors[3][:,i], label=f'comp{i}')
    ax.set(xlabel="Hours a Day", ylabel="Factor value")
leg = ax.legend();

# plot Day components
plt.xticks(ticks = list(range(0, 30)), labels = list(range(1, 31)))
for i in range(7):
    plt.plot(factors[2][:,i])

x = np.linspace(1, 31, 30)
fig, ax = plt.subplots()
for i in range(7):
    ax.plot(x, factors[2][:,i], label=f'comp{i}')
    ax.set(xlabel="Days in JAN", ylabel="Factor value")
leg = ax.legend();

# plot origin pattern components
for i in range(15):
    plt.plot(factors[0][:,i])

x = np.linspace(0, 98, 98)
fig, ax = plt.subplots()
for i in range(4):
    ax.plot(x, factors[0][:,i], label=f'comp{i}')
    ax.set(xlabel="Origin STN", ylabel="Factor value")
leg = ax.legend();

# plot destination pattern components
for i in range(15):
    plt.plot(factors[1][:,i])

x = np.linspace(0, 98, 98)
fig, ax = plt.subplots()
for i in range(4):
    ax.plot(x, factors[1][:,i], label=f'comp{i}')
    ax.set(xlabel="Destination STN", ylabel="Factor value")
leg = ax.legend();














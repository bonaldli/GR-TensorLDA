# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:12:13 2020

@author: zlibn
"""

import numpy as np
import matplotlib.pyplot as plt
alpha = [10, 5, 3, 1] # k=4, 4 topics 
s = np.random.dirichlet(alpha, size = 10).transpose()

plt.barh(range(10), s[0])
plt.barh(range(10), s[1], left=s[0], color='g')
plt.barh(range(10), s[2], left=s[0]+s[1], color='r')
plt.barh(range(10), s[3], left=s[0]+s[1]+s[2], color='k')
plt.title("Lengths of Strings")

num = 10
z = np.random.multinomial(num, s[:,0], size = 1) # we throw a dice 10 times


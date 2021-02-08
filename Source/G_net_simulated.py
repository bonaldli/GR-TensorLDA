# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
G_net= np.random.choice([0, 1], size=(98, 98), p=[9./10, 1./10])
for i in range(98):
    for j in range(max(0,i-2), min(i+3, 98)):
        G_net[i,j] = 1
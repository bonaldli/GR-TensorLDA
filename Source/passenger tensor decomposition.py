# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:57:03 2020

@author: zlibn
"""
# In[] Import Neccessary Packages
import tensorly as tl 
from tensorly.decomposition import non_negative_tucker
import matplotlib.pyplot as plt
import numpy as np

user_tensor = user100_tensor

# In[] 
# Experiment 1: Non-negative Tucker Decomposition
core, factors = non_negative_tucker(user_tensor, rank=(20, 10, 10, 3), n_iter_max=3000)

reconstruction = tl.tucker_to_tensor(core, factors)
error = tl.norm(reconstruction - user_tensor)/tl.norm(user_tensor)
print(error)

user0 = user_tensor[0,:,:,:]
user0_recon = reconstruction[0,:,:,:]
user0_recon = np.round(user0_recon, 4)

# In[] 
# Experiment 2: Robust PCA
#### 2.1 Generate boolean mask

mask = user_tensor > 0
mask = mask.astype(np.int)

#### 2.2 Decomposition
#### Decomposes a tensor X into the sum of a low-rank component D and a sparse component E.
from tensorly.decomposition import robust_pca
D, E = robust_pca(X = user_tensor, mask = mask, n_iter_max = 3000)
D0 = D[0,:,:,:]
E0 = E[0,:,:,:]

# In[] 
# Experiment 3: Bayesian Tensor Completion
#### 3.1 Prepare for Matlab
import scipy.io
scipy.io.savemat('C:/Users/zlibn/Desktop/user_tensor.mat', mdict={'user_tensor': user_tensor})
scipy.io.savemat('C:/Users/zlibn/Desktop/mask.mat', mdict={'mask': mask})



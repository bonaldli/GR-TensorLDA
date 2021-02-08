# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:20:10 2020

@author: zlibn
"""

def update_beta_w_graph_v2(lam, beta_no_g, mu, G_net, G_poi):
    
    num_topic = beta_no_g.shape[0]
    num_word = beta_no_g.shape[1]
    GRAD_THRESH = 0.001
    iteration = 0
    
    # Initialization
    count_zw, count_z = initial_count(num_topic, num_word)
    beta_old = update_beta(count_zw, count_z)
    
    beta_old_exp = np.exp(beta_old)
    beta_no_g_exp = np.exp(beta_no_g)
    
    #beta_w_g = np.exp(beta_old)
    beta_w_g_exp = np.zeros((num_topic, num_word)) # beta with graph, real value!
    grad_j_max = 10
    beta_w_g_z_norm_old = 0.1
    #gradient_norm_old = 100
    converged = 10
    #converged_grad = 10
    gradient = np.zeros((num_topic, num_word))
    hessian = np.zeros((num_topic, num_word, num_word))
    for z in range(num_topic):
        while grad_j_max > GRAD_THRESH and iteration < MAX_NT_ITER: #all(math.fabs(df)> NEWTON_THRESH for df in d_beta_j) == True and 
            iteration += 1
            # Calculate gradient
            d_beta_j = d_blhood(beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi) # R^(V * 1), real value!
            # Calculate Hessian
            d2_beta = d2_blhood(beta_old_exp, z, lam, beta_no_g_exp, mu, G_net, G_poi) # R^(V * V), real value!
            # Update beta_w_g_exp
            beta_w_g_exp[z,:] = beta_old_exp[z,:] - np.dot(np.linalg.inv(d2_beta), d_beta_j) # real value!
            # Convergence index
            grad_j_max = max(d_beta_j)
            beta_w_g_z_norm = np.linalg.norm(beta_w_g_exp[z,:])
            converged = math.fabs(beta_w_g_z_norm -beta_w_g_z_norm_old) / beta_w_g_z_norm_old
            
            # update beta_old_exp used for next iteration
            beta_old_exp[z,:] = beta_w_g_exp[z,:]
            beta_w_g_z_norm_old = beta_w_g_z_norm
            
            MTRobot.sendtext(worker_idx, f' Topic {z} NT iter{iteration}  grad max:{grad_j_max:.5f}  betaz: {beta_w_g_z_norm:.5f} betacon: {converged:.5f}')
        gradient[z,:] = d_beta_j
        hessian[z,:,:] = d2_beta
    beta_w_g_exp[beta_w_g_exp <= 0] = 1e-2 # to aviod non-feasible value
    beta_w_g_exp = normalize(beta_w_g_exp, norm='l1')
    beta_w_g = np.log(beta_w_g_exp)
    return beta_w_g, gradient, hessian # log value

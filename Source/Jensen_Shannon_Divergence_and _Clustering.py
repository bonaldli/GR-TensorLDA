# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:16:09 2021

@author: zlibn
"""
import numpy as np
from scipy.spatial import distance

def distance_matrix(x):
    num_user = x.shape[0]
    result = np.zeros((num_user, num_user))
    for i in range(num_user):
        for j in range(i, num_user):
            result[i,j] = distance.jensenshannon(x[i],x[j])
    result = result + result.T - np.diag(np.diag(result)) # Copy upper triangle to lower triangle in a python matrix
    return result

theta = theta_lam01
num_user = theta.shape[0]
JKL = theta.shape[1] * theta.shape[2] * theta.shape[3]
theta_matrix = np.zeros((num_user, JKL))
for u in range(num_user):
    temp = theta[u,:,:,:]
    temp_f = temp.flatten('F')
    theta_matrix[u,:] = temp_f

JS_matrix = distance_matrix(theta_matrix)


from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(affinity='precomputed', n_clusters=10, linkage='complete').fit(JS_matrix)
cluster_assignment = model.labels_

# only choose certain cluster
n_clusters=10
cluster_summary = np.zeros((n_clusters,2))
for c in range(n_clusters):
    member_index = np.where(cluster_assignment == c)
    cluster_summary[c,0] = c
    cluster_summary[c,1] = len(member_index[0])
    
theta_cluster9 = theta[np.where(cluster_assignment == 9)[0],:,:,:] 
# In[] Visualization
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
#model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
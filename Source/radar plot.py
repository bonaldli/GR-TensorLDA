# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:26:59 2020

@author: zlibn
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from matplotlib.pyplot import cm


zd_poi_n_df = pd.DataFrame(zd_poi_n, columns=['Hotel', 'Leisure Shopping', 'Major Buildings', 'Public Facilities', 'Residential', 'School', 'Public Transport'])
# Set data
df = zd_poi_n_df

# ------- PART 1: Create background
 
# number of variable
categories=list(df)[0:]
num_categories = len(categories)

num_topic = df.shape[0]
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
angles += angles[:1]


fig = plt.figure(figsize=(6,6))
fig.show()
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, size=12)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.1, 0.2, 0.3], ["0.1","0.2", "0.3"], color="grey", size=12)
plt.ylim(0,0.3)


#values=df.loc[0].values.flatten().tolist()
#values += values[:1]
#ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'origin topic {0}')
#ax.fill(angles, values, 'b' , alpha=0.1)
color=iter(cm.rainbow(np.linspace(0,1,num_topic)))
for z in range(num_topic):
    c = next(color)
    values=df.loc[z].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'origin topic {z}')
    #ax.fill(angles, values, c , alpha=0.1)

# Add legend
plt.legend(loc='best', bbox_to_anchor=(0.1, 0.1))
plt.draw()
plt.savefig("C:\\Users\\zlibn\\Desktop\\"+"destination_poi.png", dpi = (300))

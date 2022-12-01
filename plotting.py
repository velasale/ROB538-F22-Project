# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:20:30 2022

@author: nigel
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

with open("CF_1_data.pkl",'rb') as f1:
    cf1_data = pkl.load(f1)
    

with open("Local_data.pkl",'rb') as f2:
    local_data = pkl.load(f2)

with open("C2_data.pkl",'rb') as f3:
    cf2_data = pkl.load(f3)
    
smoothness = 20
    
local_rewards = local_data['Global Reward']

local_rewards = np.array([i[-1] for i in local_rewards])

local_rewards = moving_average(local_rewards, smoothness)

cf1_rewards = cf1_data['Global Reward']

cf1_rewards = np.array([i[-1] for i in cf1_rewards])


cf1_rewards = moving_average(cf1_rewards, smoothness)

cf2_rewards = cf2_data['Global Reward']

cf2_rewards = np.array([i[-1] for i in cf2_rewards])

cf2_rewards = moving_average(cf2_rewards, smoothness)

# plt.plot(range(len(local_rewards)),local_rewards)
# plt.plot(range(len(cf1_rewards)),cf1_rewards)
# plt.plot(range(len(cf2_rewards)),cf2_rewards)
# plt.legend(['Local Reward Only','Counterfactual 1','Countfactual 2'])
# plt.xlabel('Episode')
# plt.ylabel('Average Global Reward')
# plt.title('Smoothed Global Reward (x20)')
# plt.show()

with open("D1_Static_2data.pkl",'rb') as f1:
    cf1_data = pkl.load(f1)
    

with open("Local_Static_data.pkl",'rb') as f2:
    local_data = pkl.load(f2)

with open("D2_Static_2data.pkl",'rb') as f3:
    cf2_data = pkl.load(f3)
    
smoothness = 20
    
local_rewards = local_data['Global Reward']

local_rewards = np.array([i[-1] for i in local_rewards])

local_rewards = moving_average(local_rewards, smoothness)

cf1_rewards = cf1_data['Global Reward']

cf1_rewards = np.array([i[-1] for i in cf1_rewards])


cf1_rewards = moving_average(cf1_rewards, smoothness)

cf2_rewards = cf2_data['Global Reward']

cf2_rewards = np.array([i[-1] for i in cf2_rewards])

cf2_rewards = moving_average(cf2_rewards, smoothness)

plt.plot(range(len(local_rewards)),local_rewards)
plt.plot(range(len(cf1_rewards)),cf1_rewards)
plt.plot(range(len(cf2_rewards)),cf2_rewards)
plt.legend(['Local Reward Only','Difference Reward','Difference Reward + Next State'])
plt.xlabel('Episode')
plt.ylabel('Average Global Reward')
plt.title('Smoothed Global Reward (x20)')
plt.show()
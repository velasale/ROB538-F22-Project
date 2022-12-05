import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


smoothness = 50
with open("global1_pathfinding_1253.pkl", 'rb') as f2:
    global1_r = pkl.load(f2)

with open("global2_pathfinding_1253.pkl", 'rb') as f2:
    global2_r = pkl.load(f2)

with open("globallocal_pathfinding_1253.pkl", 'rb') as f2:
    global3_r = pkl.load(f2)

with open("diffonly_pathfinding_1253.pkl", 'rb') as f2:
    diffonly_r = pkl.load(f2)

with open("diff1_pathfinding_1253.pkl", 'rb') as f2:
    diff_r = pkl.load(f2)

with open("difflocal_pathfinding_1253.pkl", 'rb') as f2:
    difflocal_r = pkl.load(f2)

with open("local_pathfinding_1253.pkl", 'rb') as f3:
    local_r = pkl.load(f3)

difflocal_r = difflocal_r['total']

difflocal_r = np.array(difflocal_r)

difflocal_r = moving_average(difflocal_r, smoothness)

global3_r = global3_r['total']

global3_r = np.array(global3_r)

global3_r = moving_average(global3_r, smoothness)


global2_r = global2_r['total']

global2_r = np.array(global2_r)

global2_r = moving_average(global2_r, smoothness)


global1_r = global1_r['total']

global1_r = np.array(global1_r)

global1_r = moving_average(global1_r, smoothness)


diffonly_r = diffonly_r['total']

diffonly_r = np.array(diffonly_r)

diffonly_r = moving_average(diffonly_r, smoothness)


local_r = local_r['total']

local_r = np.array(local_r)

local_r = moving_average(local_r, smoothness)


diff_r = diff_r['total']

cf2_rewards = np.array(diff_r)

diff_r = moving_average(diff_r, smoothness)

plt.plot(range(len(diff_r)), diff_r)
plt.plot(range(len(difflocal_r)), difflocal_r)
plt.plot(range(len(diffonly_r)), diffonly_r)
plt.plot(range(len(global1_r)), global1_r)
plt.plot(range(len(global3_r)), global3_r)
plt.plot(range(len(global2_r)), global2_r)
plt.plot(range(len(local_r)), local_r)
plt.legend(
    ["Difference Reward + Local (Slight Shaping)", "Difference Reward + Local (No Shaping)", "Difference Reward",
     "Global Reward 1 (%Completed + %tasks_revealed)", "Global Reward 1 + Local Reward", "Global Reward 2 (%Completed)",
     "Local Reward"])
plt.xlabel('Episode')
plt.ylabel('Average Global Reward')
plt.title('Smoothed Global Reward (x50)')
plt.show()

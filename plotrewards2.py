import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


smoothness = 100
# SMALL 2
with open("global1_pathfinding_1253.pkl", 'rb') as f2:
    global_r = pkl.load(f2)

# with open("difflocal_pathfinding_1253.pkl", 'rb') as f2:
#     diff_r = pkl.load(f2)

# with open("local_pathfinding_1253.pkl", 'rb') as f3:
#     local_r = pkl.load(f3)

# LARGE 6
with open("localsmall16_6_5000_pathfinding_1253.pkl", 'rb') as f2:
    diff_r = pkl.load(f2)

with open("diffsmall16_6_5000_pathfinding_1253.pkl", 'rb') as f2:
    local_r = pkl.load(f2)

# with open("globalsmall16_6_5000_pathfinding_1253.pkl", 'rb') as f2:
#    global_r = pkl.load(f2)

# SMALL 8 with 4 agents
with open("difforig_small8_4_pathfinding_1253.pkl", 'rb') as f2:
    diff_r = pkl.load(f2)

with open("localorig_small8_4_pathfinding_1253.pkl", 'rb') as f2:
    local_r = pkl.load(f2)

# LARGE 4
# with open("diffsmall16_4_5000_pathfinding_1253.pkl", 'rb') as f2:
#     diff_r = pkl.load(f2)

# with open("localsmall16_4_5000_pathfinding_1253.pkl", 'rb') as f2:
#     local_r = pkl.load(f2)

# with open("globalsmall16_4_5000_pathfinding_1253.pkl", 'rb') as f2:
#     global_r = pkl.load(f2)


local_r = local_r['total']

local_r = np.array(local_r)
local_r_all = np.array(local_r)

local_r = moving_average(local_r, smoothness)


diff_r = diff_r['total']

diff_r = np.array(diff_r)
diff_r_all = np.array(diff_r)

diff_r = moving_average(diff_r, smoothness)


global_r = global_r['total']

global_r = np.array(global_r)
global_r_all = np.array(global_r)

global_r = moving_average(global_r, smoothness)


plt.plot(range(len(diff_r)), diff_r, color="blue", alpha=1, linewidth=6)
plt.plot(range(len(local_r)), local_r, color="orange",  alpha=1, linewidth=6)
plt.plot(range(len(global_r)), global_r, alpha=1, color="black", linewidth=6)
plt.plot(diff_r_all, color="blue", alpha=0.2, label='Total Percent')
plt.plot(local_r_all, color="orange", alpha=0.2, label='Total Percent')
plt.plot(global_r_all, color="black", alpha=0.2, label='Total Percent')
plt.xticks(fontsize=40)
plt.yticks(fontsize=40, ticks=np.arange(0, 101, 20))
ax = plt.gca()
vals = ax.get_yticks()
ax.set_yticklabels(['{:.0f}%'.format(x) for x in vals])
#     ["Difference Reward + Local (Slight Shaping)", "Difference Reward + Local (No Shaping)", "Difference Reward",
#      "Global Reward 1 (%Completed + %tasks_revealed)", "Global Reward 1 + Local Reward", "Global Reward 2 (%Completed)",
#      "Local Reward"])
# plt.legend(["Difference Reward + Local (Slight Shaping)", "Difference Reward + Local (No Shaping)",
#            "Local Reward", "Difference Reward", "Global Reward 2 (%Completed)"])

plt.legend(["Difference Reward", " Local Reward", "Global Reward"], prop={'size': 36})
plt.xlabel('Episode', fontsize=44)
plt.ylabel('Global Reward', fontsize=44)
#plt.title('2 Agents, Small Map', fontsize=46)
#plt.title('4 Agents, Large Map', fontsize=46)
plt.grid(True)
plt.show()

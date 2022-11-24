import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import namedtuple, deque
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

L1 = 50
L2 = 100
L3 = 80
L4 = 16
BATCH_SIZE = 50


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, device) -> None:
        super(DQN, self).__init__()
        self.flatten = nn.Flatten
        self.memory = ReplayMemory(10000)
        self.eps = .2
        self.episode_count = 1
        self.global_reward = []
        self.device = device
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(L1, L2),
            nn.ReLU(),
            nn.Linear(L2, L3),
            nn.ReLU(),
            nn.Linear(L3, L4)
        )
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.linear_relu_stack.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = x.to(self.device)
        logits = self.linear_relu_stack(x)
        return logits

    def plot_global_reward(self):
        plt.figure(2)
        plt.clf()
        rewards_t = torch.tensor(self.global_reward, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Global Reward')
        plt.plot(rewards_t.numpy())
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

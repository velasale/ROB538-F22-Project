# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:49:26 2022

@author: Kegan and Nigel and whoever else worked on this
"""

from collections import deque
from dataclasses import dataclass, asdict
import dataclasses
from abc import ABC, abstractmethod
import json
import logging
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pickle as pkl
from queue import PriorityQueue, Queue
from collections import deque

# Why do we save things as timesteps when there aren't any methods? Why not save as a dictionary since we need to make it a dictionary at some point down the line?


class ReplayBuffer:
    def __init__(self, buffer_size: int = 50000, no_delete=False):
        '''
        Constructor takes in the necessary state, action, reward objects and sets the parameters of the 
        replay buffer. Currently only even sampling is supported by this replay buffer (no weighted transitions)
        but it could be added fairly easily. 
        :param buffer_size: Size of the deque before oldest entry is deleted.
        :param no_delete: Set if you do not wish old entries to be deleted. 
        :param state: State object.
        :param action: action object.
        :param reward: Reward object.
        :type buffer_size: int
        :type no_delete: bool
        :type state: :func:`~mojograsp.simcore.state.State` 
        :type action: :func:`~mojograsp.simcore.action.Action` 
        :type reward: :func:`~mojograsp.simcore.reward.Reward` 
        '''
        self.buffer_size = buffer_size
        self.prev_timestep = None

        # Using list instead of deque for access speed and forward rollout access
        self.buffer = deque(maxlen=buffer_size)
        #self.buffer = []

    def update_buffer(self, episode_num: int, timestep_num: int, state: list, action: list, reward: list,
                      next_state: list):
        """
        Method adds a timestep using the state, action and reward get functions given the episode_num and
        timestep_num from the Simmanager. 
        :param episode_num: Episode number.
        :param timestep_num: Timestep number.
        :type episode_num: int
        :type timestep_num: int
        """
        tstep = {'episode': episode_num, 'timestep': timestep_num, 'state': state,
                 'action': action, 'reward': reward, 'next_state': next_state}
        self.buffer.append(tstep)

    def update_buffer_list(self, tstep_list, opposite_state):
        """
        Method adds a timestep using the state, action and reward get functions given the episode_num and
        timestep_num from the Simmanager. 
        :param episode_num: Episode number.
        :param timestep_num: Timestep number.
        :type episode_num: int
        :type timestep_num: int
        """
        for i in range(len(tstep_list)):
            coordinates = tstep_list[i]["state"][-2:]
            tstep_list[i]["state"] = opposite_state + coordinates
            self.buffer.append(tstep_list[i])

    def update_buffer_shared(self, episode_num: int, timestep_num: int, state: list, action: list, reward: list,
                             next_state: list):
        """
        Method adds a timestep using the state, action and reward get functions given the episode_num and
        timestep_num from the Simmanager. 
        :param episode_num: Episode number.
        :param timestep_num: Timestep number.
        :type episode_num: int
        :type timestep_num: int
        """
        tstep = {'episode': episode_num, 'timestep': timestep_num, 'state': state,
                 'action': action, 'reward': reward, 'next_state': next_state}
        self.buffer.append(tstep)
        return tstep

    def update_buffer_rollback_reward(
            self, episode_num: int, rollback: int, rollback_reward: int, rollback_decay: float):
        """
        Method adds a timestep using the state, action and reward get functions given the episode_num and
        timestep_num from the Simmanager. 
        :param episode_num: Episode number.
        :param timestep_num: Timestep number.
        :type episode_num: int
        :type timestep_num: int
        """
        if len(self.buffer) > rollback:
            for i in range(rollback):
                if self.buffer[-(i+1)]["reward"] == 10 or self.buffer[-i]["episode"] != episode_num:
                    break
                else:
                    self.buffer[-(i+1)]["reward"] += rollback_reward * \
                        (rollback_decay * (i+1))

    def update_buffer_rollback_reward_shared(
            self, episode_num: int, rollback: int, rollback_reward: int, rollback_decay: float):
        """
        Method adds a timestep using the state, action and reward get functions given the episode_num and
        timestep_num from the Simmanager. 
        :param episode_num: Episode number.
        :param timestep_num: Timestep number.
        :type episode_num: int
        :type timestep_num: int
        """
        shared = []
        if len(self.buffer) > rollback:
            for i in range(rollback):
                if self.buffer[-(i+1)]["reward"] == 10 or self.buffer[-i]["episode"] != episode_num:
                    break
                else:
                    self.buffer[-(i+1)]["reward"] += rollback_reward * \
                        (rollback_decay * (i+1))
                    shared.append(self.buffer[-(i+1)])
        return shared

    def save_buffer(self, file_path: str = None):
        """
        Method saves the current replay buffer to a json file at the location of the given file_path.
        :param file_path: desired destination and name of the json file.
        :type file_path: str
        """
        with open(file_path, 'wb') as fout:
            pkl.dump(self.buffer, fout)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return(len(self.buffer))

    def get_average_reward(self, num):
        rewards = self.buffer[-num:]
        avg_reward = 0
        for r in rewards:
            avg_reward += r['reward']
        avg_reward = avg_reward/len(rewards)
        return avg_reward

    def get_min_reward(self, num):
        rewards = self.buffer[-num:]
        reward2 = []
        for r in rewards:
            reward2.append(r['reward'])
        min_reward = min(reward2)
        return min_reward

    def get_max_reward(self, num):
        rewards = self.buffer[-num:]
        reward2 = []
        for r in rewards:
            reward2.append(r['reward'])
        max_reward = max(reward2)
        return max_reward

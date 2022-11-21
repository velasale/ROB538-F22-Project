# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:48:50 2022

@author: nigel
"""

import torch
import numpy as np
from learning import DiscreteLearning, LearningBaseClass

class AgentBaseClass():
    def __init__(self, field_size, num_agent_type=2):
        # memory has shape hxwx2. self.memory[:,:,0] is the state of the system
        # self.memory[:,:,1] is the timestep associated with that state information
        # so that we can give both agents the most updated information when they
        # synchronize their information
        # 0 = empty
        # 1 = us
        # 2 = empty tree
        # 3 = tree with apple
        # 4 = tree with branch
        # 5 = tree with branch and apple
        # 6 = other robot
        # 7 = unseen
        
        #start location assumed to be at 0,0
        self.memory = np.zeros([field_size[0], field_size[1],2])
        self.memory[0,0,0] = 1
        self.policy = LearningBaseClass()
        self.field_size = field_size
        
    def apply_sensor(self, sensor_reading, sensor_start, timestep):
        # here we assume that the sensor reading is a square,
        # smallest corner at sensor start
        sensor_size = len(sensor_reading)
        self.memory[sensor_start[0]:sensor_start[0]+sensor_size,sensor_start[1]:sensor_start[1]+sensor_size,1] = timestep
        self.memory[sensor_start[0]:sensor_start[0]+sensor_size,sensor_start[1]:sensor_start[1]+sensor_size,0]
        
    def update_map(self, other_map):
        #probably a much more efficient way to do this 
        # where we synchronize two maps
        for i in range(self.field_size[0]):
            for j in range(self.field_size[1]):
                if self.memory[i,j,1] > other_map[i,j,1]:
                    self.memory[i,j,:] = other_map[i,j,:]
    
    def select_action(self):
        return self.policy.select_action(self.memory[:,:,0])
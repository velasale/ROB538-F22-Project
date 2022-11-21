# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:13:50 2022

@author: nigel
"""

# import Markers
import copy
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simple_normalize(x_tensor):
    # order is pos, orientation (quaternion), joint angles, velocity
    maxes = torch.tensor([0.2, 0.35, 0.1, 1, 1, 1, 1, np.pi/2, 0, np.pi/2, np.pi, 1, 1, 1, 0.5, 0.5]).to(device)
    mins = torch.tensor([-0.2, -0.05, 0.0, 0, 0, 0, 0,-np.pi/2, -np.pi, -np.pi/2, 0, 0, 0, 0, -0.01, -0.01]).to(device)
    y_tensor = (x_tensor-mins)/(maxes-mins)
    return y_tensor
    

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=60
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=60, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed
    
    def encode(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        return code

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, action_dim)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.max_action = max_action

    def forward(self, state):
        # print("State Actor: {}\n{}".format(state.shape, state))
        state = simple_normalize(state)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.nn.functional.normalize(self.l3(a))

# OLD PARAMS WERE 400-300, TESTING 100-50
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.leaky = nn.LeakyReLU()
        self.l1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.max_q_value = 0.1

    def forward(self, state):
        # print('input to critic', torch.cat([state, action], -1))
        state = simple_normalize(state)
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        # print("Q Critic: {}".format(q))
        q = torch.tanh(self.l3(q))
        return q * self.max_q_value


class DDPGfD():
    def __init__(self, arg_dict: dict = None, TensorboardName = None):
        # state_path=None, state_dim=32, action_dim=4, max_action=1.57, n=5, discount=0.995, tau=0.0005, batch_size=10,
        #          expert_sampling_proportion=0.7):
        if arg_dict is None:
            print('no arg dict')
            arg_dict = {'state_dim': 32, 'action_dim': 5, 'max_action': 1.57, 'n': 5, 'discount': 0.995, 'tau': 0.005,
                        'batch_size': 10, 'expert_sampling_proportion': 0.7}
        self.state_dim = arg_dict['state_dim']
        self.action_dim = arg_dict['action_dim']
        self.actor = Actor(self.state_dim, self.action_dim, arg_dict['max_action']).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-4)

        self.actor_loss = []
        self.critic_loss = []
        self.critic_L1loss = []
        self.critic_LNloss = []
        if TensorboardName is None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter('runs/'+TensorboardName)

        self.discount = arg_dict['discount']
        self.tau = arg_dict['tau']
        self.n = arg_dict['n']
        self.network_repl_freq = 2
        self.total_it = 0
        self.lambda_Lbc = 1

        # Most recent evaluation reward produced by the policy within training
        self.avg_evaluation_reward = 0

        self.batch_size = arg_dict['batch_size']
        self.rng = default_rng()
        self.rollout = True
        self.u_count = 0
        
        #self.encoder = AE(input_shape=400)
        #might use later but first try not

    def select_action(self, state):
        state = torch.FloatTensor(np.reshape(state, (1, -1))).to(device)
        # print("TYPE:", type(state), state)
        action = self.actor(state).cpu().data.numpy().flatten()
        # print("Action: {}".format(action))
        return action

    def grade_action(self, state, action):
        state = torch.FloatTensor(np.reshape(state, (1,-1))).to(device)
        action = torch.FloatTensor(np.reshape(action, (1,-1))).to(device)
        grade = self.critic(state, action).cpu().data.numpy().flatten()
        return grade

    def copy(self, policy_to_copy_from):
        """ Copy input policy to be set to another policy instance
		policy_to_copy_from: policy that will be copied from
        """
        # Copy the actor and critic networks
        self.actor = copy.deepcopy(policy_to_copy_from.actor)
        self.actor_target = copy.deepcopy(policy_to_copy_from.actor_target)
        self.actor_optimizer = copy.deepcopy(policy_to_copy_from.actor_optimizer)

        self.critic = copy.deepcopy(policy_to_copy_from.critic)
        self.critic_target = copy.deepcopy(policy_to_copy_from.critic_target)
        self.critic_optimizer = copy.deepcopy(policy_to_copy_from.critic_optimizer)

        self.discount = policy_to_copy_from.discount
        self.tau = policy_to_copy_from.tau
        self.n = policy_to_copy_from.n
        self.network_repl_freq = policy_to_copy_from.network_repl_freq
        self.total_it = policy_to_copy_from.total_it
        self.lambda_Lbc = policy_to_copy_from.lambda_Lbc
        self.avg_evaluation_reward = policy_to_copy_from.avg_evaluation_reward

        # Sample from the expert replay buffer, decaying the proportion expert-agent experience over time
        self.initial_expert_proportion = policy_to_copy_from.initial_expert_proportion
        self.current_expert_proportion = policy_to_copy_from.current_expert_proportion
        self.sampling_decay_rate = policy_to_copy_from.sampling_decay_rate
        self.sampling_decay_freq = policy_to_copy_from.sampling_decay_freq
        self.batch_size = policy_to_copy_from.batch_size

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        np.save(filename + "avg_evaluation_reward", np.array([self.avg_evaluation_reward]))

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

    def update_target(self):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def calc_roll_rewards(self,rollout_reward):
        sum_rewards = []
        num_rewards = []
        for reward_set in rollout_reward:
            temp_reward = 0
            for count, step_reward in enumerate(reward_set):
                temp_reward += step_reward*self.discount**count
            sum_rewards.append(temp_reward)
            num_rewards.append(count+1)
        num_rewards = torch.tensor(num_rewards).to(device)
        num_rewards = torch.unsqueeze(num_rewards,1)
        sum_rewards = torch.tensor(sum_rewards)
        sum_rewards = torch.unsqueeze(sum_rewards,1)
        return sum_rewards, num_rewards

    def collect_batch(self, replay_buffer):
        pass
    

    def train(self, state, action, reward, next_state):
        """ Update policy based on full trajectory of one episode """
        #next state holds ALL POSSIBLE NEXT STATES from current state (not including other agent's moves)
        self.total_it += 1
        with torch.no_grad: 
            action_probabilities = self.actor(state)
        
        target_Q = (self.critic_target(next_state) * action_probabilities).sum()

        target_Q = reward + (self.discount * target_Q).detach()  # bellman equation

        target_Q = target_Q.float()

        # Get current Q estimate
        current_Q = self.critic(state)

        # L_1 loss (Loss between current state, action and reward, next state, action)
        critic_L1loss = F.mse_loss(current_Q, target_Q)

        # L_2 loss (Loss between current state, action and reward, n state, n action)

        # Total critic loss
        critic_loss = critic_L1loss.float()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(next_state) * self.actor(state)
        actor_loss = actor_loss.mean()
        # input(actor_loss)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.writer.add_scalar('Loss/critic',critic_loss.detach(),self.total_it)
        self.writer.add_scalar('Loss/actor',actor_loss.detach(),self.total_it)


        # update target networks
        if self.total_it % self.network_repl_freq == 0:
            self.update_target()
            self.u_count +=1
            # print('updated ', self.u_count,' times')
            # print('critic of state and action')
            # print(self.critic(state, self.actor(state)))
            # print('target q - current q')
            # print(target_Q-current_Q)
        return actor_loss.item(), critic_loss.item()



def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    :param: long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr
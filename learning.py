# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:56:55 2022

@author: nigel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import time
from replaybuffer import ReplayBuffer
import copy
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LearningBaseClass():
    def __init__(self):
        pass

    def select_action(self, state):
        return 0

    def update_epsilon(self):
        pass

    def update_weight(self, state, action, reward):
        pass


class DiscreteLearning():
    def __init__(self, q_table_size):
        # this implementation only works for local shit
        self.qtable = torch.rand([3, 3, 3, 3, 3, 3, 3, 3, 5])
        self.gamma = 0.4
        self.learning_rate = 0.1
        self.action_key = ['left', 'right', 'up', 'down', 'interact']
        self.count = 0

    def select_action(self, state, more_state, possible_actions, action_keys, cur_pos):
        valid_inds = [[cur_pos[0] - 1, cur_pos[1] - 1],
                      [cur_pos[0] - 1, cur_pos[1]],
                      [cur_pos[0] - 1, cur_pos[1] + 1],
                      [cur_pos[0], cur_pos[1] - 1],
                      [cur_pos[0], cur_pos[1] + 1],
                      [cur_pos[0] + 1, cur_pos[1] - 1],
                      [cur_pos[0] + 1, cur_pos[1]],
                      [cur_pos[0] + 1, cur_pos[1]+1]]
        state_for_table = [0]*8
        l_state = state.tolist()
        m_state = more_state.tolist()
        for i, ind in enumerate(valid_inds):
            if -1 in ind:
                state_for_table[i] = 2
            elif ind in l_state:
                temp = int(m_state[l_state.index(ind)])
                if temp == 1 or temp == 3:
                    state_for_table[i] = 1
                elif temp == 2 or temp == 4 or temp == -10:
                    state_for_table[i] = 2
                else:
                    state_for_table[i] = 0
        # yes i know its nasty
        agent_actions = self.qtable[state_for_table[0]][state_for_table[1]][state_for_table[2]][state_for_table[3]][
            state_for_table[4]][state_for_table[5]][state_for_table[6]][state_for_table[7]]

        actions = np.argsort(agent_actions).detach().tolist()
        for action in actions[::-1]:
            if self.action_key[action] in action_keys:
                indexthing = action_keys.index(self.action_key[action])
                return possible_actions[indexthing], self.action_key[action]
            else:
                self.count += 1
                self.qtable[state_for_table[0]][state_for_table[1]][state_for_table[2]][state_for_table[3]][
                    state_for_table[4]][state_for_table[5]][state_for_table[6]][state_for_table[7]][action] = -10

    def select_action_apple(self, state_for_table, possible_actions, action_keys):
        # yes i know its nasty
        agent_actions = self.qtable[state_for_table[0]][state_for_table[1]][state_for_table[2]][state_for_table[3]][
            state_for_table[4]][state_for_table[5]][state_for_table[6]][state_for_table[7]]

        actions = np.argsort(agent_actions).detach().tolist()
        for action in actions[::-1]:
            if self.action_key[action] in action_keys:
                indexthing = action_keys.index(self.action_key[action])
                return possible_actions[indexthing], self.action_key[action]
            else:
                self.count += 1
                self.qtable[state_for_table[0]][state_for_table[1]][state_for_table[2]][state_for_table[3]][
                    state_for_table[4]][state_for_table[5]][state_for_table[6]][state_for_table[7]][action] = -10

    def update_epsilon(self):
        self.epsilon *= 0.999

    def get_max(self, state):
        a = np.argmax(self.qtable[state])
        return a[0]

    def train_apple(self, init_state, action, reward, init_next_state):
        action_ind = self.action_key.index(action)
        self.qtable[init_state[0]][init_state[1]][init_state[2]][init_state[3]][init_state[4]][init_state[5]][
            init_state[6]][init_state[7]][action_ind] += self.learning_rate * (reward + self.gamma * torch.max(
                self.
                qtable[init_next_state[0]][init_next_state[1]
                                           ][init_next_state[2]][init_next_state[3]]
                [init_next_state[4]][init_next_state[5]][init_next_state[6]][init_next_state[7]][:]) - self.
            qtable[init_state[0]][init_state[1]][init_state[2]
                                                 ][init_state[3]][init_state[4]][init_state[5]]
            [init_state[6]][init_state[7]][action_ind])

    def update_weight(self, state, state_val, cur_pos, action, reward, next_state, next_state_val, next_pos):
        # fix the state stuff so that its like we expect it
        # both for state and next state
        valid_inds = [[cur_pos[0] - 1, cur_pos[1] - 1],
                      [cur_pos[0] - 1, cur_pos[1]],
                      [cur_pos[0] - 1, cur_pos[1] + 1],
                      [cur_pos[0], cur_pos[1] - 1],
                      [cur_pos[0], cur_pos[1] + 1],
                      [cur_pos[0] + 1, cur_pos[1] - 1],
                      [cur_pos[0] + 1, cur_pos[1]],
                      [cur_pos[0] + 1, cur_pos[1]+1]]
        valid_second_inds = [[next_pos[0] - 1, next_pos[1] - 1],
                             [next_pos[0] - 1, next_pos[1]],
                             [next_pos[0] - 1, next_pos[1] + 1],
                             [next_pos[0], next_pos[1] - 1],
                             [next_pos[0], next_pos[1] + 1],
                             [next_pos[0] + 1, next_pos[1] - 1],
                             [next_pos[0] + 1, next_pos[1]],
                             [next_pos[0] + 1, next_pos[1]+1]]
        init_state = [0]*8

        l_state = state.tolist()
        m_state = state_val.tolist()
        for i, ind in enumerate(valid_inds):
            if -1 in ind:
                init_state[i] = 2
            elif ind in l_state:
                temp = int(m_state[l_state.index(ind)])
                if temp == 1 or temp == 3:
                    init_state[i] = 1
                elif temp == 2 or temp == 4:
                    init_state[i] = 2
                else:
                    init_state[i] = 0

        init_next_state = [0] * 8
        l_state = next_state.tolist()
        m_state = next_state_val.tolist()
        for i, ind in enumerate(valid_inds):
            if -1 in ind:
                init_next_state[i] = 2
            elif ind in l_state:
                temp = int(m_state[l_state.index(ind)])
                if temp == 1 or temp == 3:
                    init_next_state[i] = 1
                elif temp == 2 or temp == 4:
                    init_next_state[i] = 2
                else:
                    init_next_state[i] = 0

        action_ind = self.action_key.index(action)
        self.qtable[init_state[0]][init_state[1]][init_state[2]][init_state[3]][init_state[4]][init_state[5]][
            init_state[6]][init_state[7]][action_ind] += self.learning_rate * (reward + self.gamma * torch.max(
                self.
                qtable[init_next_state[0]][init_next_state[1]
                                           ][init_next_state[2]][init_next_state[3]]
                [init_next_state[4]][init_next_state[5]][init_next_state[6]][init_next_state[7]][:]) - self.
            qtable[init_state[0]][init_state[1]][init_state[2]
                                                 ][init_state[3]][init_state[4]][init_state[5]]
            [init_state[6]][init_state[7]][action_ind])


class ConvolutionLearning():
    def __init__(self, q_table_size):
        self.qtable = torch.rand(q_table_size)
        self.gamma = 0.9
        self.learning_rate = 1

    def select_action(self, state, possible_actions, action_keys):
        agent_actions = self.qtable[state]
        actions = np.argmax(agent_actions)
        for action, action_key in zip(actions, action_keys):
            if action in possible_actions:
                return action, action_key

    def update_epsilon(self):
        self.epsilon *= 0.99

    def get_max(self, state):
        a = np.argmax(self.qtable[state])
        return a[0]

    def update_weight(self, state, action, reward, next_state):
        self.qtable[state] += self.learning_rate*(reward + self.gamma*np.max(
            self.qtable[next_state]) - self.qtable[next_state][action])


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        print(state_dim)
        self.l1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(
            self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(
            self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, action_dim)
        torch.nn.init.kaiming_uniform_(
            self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        try:
            return torch.nn.functional.normalize(F.sigmoid(self.l3(a)))
        except IndexError:
            return torch.nn.functional.normalize(F.sigmoid(self.l3(a)), dim=0)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        print('state dim, action dim', state_dim, action_dim)
        super(Critic, self).__init__()
        self.leaky = nn.LeakyReLU()
        self.l1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(
            self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(
            self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, action_dim)
        torch.nn.init.kaiming_uniform_(
            self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, state, action_probabilities):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        q = torch.tanh(self.l3(q)) * 10
        return q * action_probabilities


class SAC():
    def __init__(self, state_dim, action_dim, TensorboardName=None):
        self.gamma = 0.9
        self.learning_rate = 1
        self.tau = 0.01
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-4, weight_decay=1e-4)
        self.action_order = ['left', 'right', 'up', 'down', 'interact']
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 20
        self.enum = 0
        self.tnum = 0
        self.total_it = 0
        self.network_repl_freq = 2
        if TensorboardName is None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter('runs/'+TensorboardName)

    def select_action(self, state, possible_actions, action_keys):
        collapsed_state = torch.tensor(state)
        collapsed_state = torch.flatten(collapsed_state)
        collapsed_state = collapsed_state.to(device)
        actions = self.actor(collapsed_state.float())
        action_prob_order = torch.argsort(actions)
        action_prob_order = action_prob_order.flip(0)
        for action in action_prob_order:
            if self.action_order[action] in action_keys:
                indexthing = action_keys.index(
                    self.action_order[action.item()])
                return possible_actions[indexthing], self.action_order[action], actions

    def update_buffer(self, state, action, reward, next_state):
        self.replay_buffer.update_buffer(
            self.enum, self.tnum, state, action, reward, next_state)
        self.tnum += 1

    def end_episode(self):
        self.tnum = 0
        self.enum += 1

    def get_max(self, state):
        a = np.argmax(self.qtable[state])
        return a[0]

    def train(self):
        if len(self.replay_buffer) > self.batch_size:
            sample = self.replay_buffer.sample(self.batch_size)
            state = []
            action = []
            reward = []
            next_state = []

            for timestep in sample:
                state.append(timestep['state'])
                action.append(timestep['action'])
                # print(type(timestep['action']))
                reward.append(timestep['reward'])
                next_state.append(timestep['next_state'])

            state = torch.tensor(state)
            state = torch.flatten(state, start_dim=1)
            state = state.to(device).float()
            action = torch.tensor(action)
            action = action.to(device).float()
            reward = torch.tensor(reward)
            reward = reward.to(device).float()

            next_state = torch.tensor(next_state)
            next_state = torch.flatten(next_state, start_dim=1)
            next_state = next_state.to(device).float()
            action_probabilities = self.actor(state)
            # print(state.shape)
            # print(next_state.shape)
            current_Q = self.critic(state, action).sum(axis=1)
            next_actions = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_actions).sum()

            # bellman equation
            target_Q = reward + (self.gamma * target_Q).detach()

            target_Q = target_Q.float()

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, action_probabilities).sum()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.writer.add_scalar(
                'Loss/critic', critic_loss.detach(), self.total_it)
            self.writer.add_scalar(
                'Loss/actor', actor_loss.detach(), self.total_it)
            self.total_it += 1

            # update target networks
            if self.total_it % self.network_repl_freq == 0:
                self.update_target()

            return actor_loss.item(), critic_loss.item()

    def update_target(self):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class SACLimited():
    def __init__(self, state_dim, action_dim, opposite_buffer, shared_buffer, TensorboardName=None):
        self.gamma = 0.9
        self.learning_rate = 1
        self.tau = 0.01
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-4, weight_decay=1e-4)
        self.action_order = ['left', 'right', 'up', 'down', 'interact']
        self.replay_buffer = ReplayBuffer()
        self.replay_buffer_shared = shared_buffer
        self.replay_buffer_opposite = opposite_buffer
        self.batch_size = 50
        self.enum = 0
        self.tnum = 0
        self.total_it = 0
        self.network_repl_freq = 2
        if TensorboardName is None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter('runs/'+TensorboardName)

    def select_action(self, tree_states, possible_actions, action_keys, cur_pos):
        collapsed_state = torch.tensor(tree_states+cur_pos)
        collapsed_state = torch.flatten(collapsed_state)
        collapsed_state = collapsed_state.to(device)
        actions = self.actor(collapsed_state.float())
        action_prob_order = torch.argsort(actions)
        action_prob_order = action_prob_order.flip(0)
        for action in action_prob_order:
            if self.action_order[action] in action_keys:
                indexthing = action_keys.index(
                    self.action_order[action.item()])
                return possible_actions[indexthing], self.action_order[action], actions

    def select_action_path(self, tree_states):
        collapsed_state = torch.tensor(tree_states)
        collapsed_state = torch.flatten(collapsed_state)
        collapsed_state = collapsed_state.to(device)
        actions = self.actor(collapsed_state.float())
        return int(torch.argmax(actions))

    def update_buffer(self, state, action, reward, next_state):
        # if reward > 0:
        #     self.replay_buffer.update_buffer_rollback_reward(
        #         self.enum, rollback=0, rollback_reward=reward, rollback_decay=0.5)
        self.replay_buffer.update_buffer(
            self.enum, self.tnum, state, action, reward, next_state)
        self.tnum += 1

    def update_buffer_shared(self, state, action, reward, next_state, other_state):
        if reward > 0:
            shared = self.replay_buffer.update_buffer_rollback_reward_shared(
                self.enum, rollback=1, rollback_reward=reward, rollback_decay=0.5)
            shared.append(self.replay_buffer.update_buffer_shared(
                self.enum, self.tnum, state, action, reward, next_state))
            self.replay_buffer_opposite.update_buffer_list(shared, other_state)

        self.replay_buffer.update_buffer(
            self.enum, self.tnum, state, action, reward, next_state)
        self.tnum += 1

    def end_episode(self):
        self.tnum = 0
        self.enum += 1

    def get_max(self, state):
        a = np.argmax(self.qtable[state])
        return a[0]

    def train(self):
        if len(self.replay_buffer) > self.batch_size * 5:
            sample = self.replay_buffer.sample(self.batch_size)
            state = [d['state'] for d in sample]
            action = [d['action'] for d in sample]
            reward = [d['reward'] for d in sample]
            next_state = [d['next_state'] for d in sample]

            # for timestep in sample:
            #     state.append(timestep['state'])
            #     # print('timestep state')
            #     # print(timestep['state'])
            #     action.append(timestep['action'])
            #     # print(type(timestep['action']))
            #     reward.append(timestep['reward'])
            #     next_state.append(timestep['next_state'])

            state = torch.tensor(state)
            state = torch.flatten(state, start_dim=1)
            state = state.to(device).float()
            action = torch.tensor(action)
            action = action.to(device).float()
            reward = torch.tensor(reward)
            reward = reward.to(device).float()

            next_state = torch.tensor(next_state)
            next_state = torch.flatten(next_state, start_dim=1)
            next_state = next_state.to(device).float()
            action_probabilities = self.actor(state)
            current_Q = self.critic(state, action).sum(axis=1)
            next_actions = self.actor_target(next_state)
            target_Q = self.critic_target(
                next_state, next_actions).max(axis=1)[0]
            # print(reward, target_Q)

            # bellman equation
            target_Q = reward + (self.gamma * target_Q).detach()

            target_Q = target_Q.float()

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, action_probabilities).sum()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.writer.add_scalar(
                'Loss/critic', critic_loss.detach(), self.total_it)
            self.writer.add_scalar(
                'Loss/actor', actor_loss.detach(), self.total_it)
            self.total_it += 1

            # update target networks
            if self.total_it % self.network_repl_freq == 0:
                self.update_target()

            return actor_loss.item(), critic_loss.item()

    def train_shared(self):
        if len(self.replay_buffer) > self.batch_size:
            if len(self.replay_buffer_shared.buffer) < self.batch_size - 40:
                sample = self.replay_buffer.sample(self.batch_size)
            else:
                sample = self.replay_buffer.sample(self.batch_size - 10)
                sample = sample + \
                    self.replay_buffer_shared.sample(self.batch_size - 40)

            state = []
            action = []
            reward = []
            next_state = []

            for timestep in sample:
                state.append(timestep['state'])
                # print('timestep state')
                # print(timestep['state'])
                action.append(timestep['action'])
                # print(type(timestep['action']))
                reward.append(timestep['reward'])
                next_state.append(timestep['next_state'])

            state = torch.tensor(state)
            state = torch.flatten(state, start_dim=1)
            state = state.to(device).float()
            action = torch.tensor(action)
            action = action.to(device).float()
            reward = torch.tensor(reward)
            reward = reward.to(device).float()

            next_state = torch.tensor(next_state)
            next_state = torch.flatten(next_state, start_dim=1)
            next_state = next_state.to(device).float()
            action_probabilities = self.actor(state)
            current_Q = self.critic(state, action).sum(axis=1)
            next_actions = self.actor_target(next_state)
            target_Q = self.critic_target(
                next_state, next_actions).max(axis=1)[0]
            # print(reward, target_Q)

            # bellman equation
            target_Q = reward + (self.gamma * target_Q).detach()

            target_Q = target_Q.float()

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, action_probabilities).sum()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.writer.add_scalar(
                'Loss/critic', critic_loss.detach(), self.total_it)
            self.writer.add_scalar(
                'Loss/actor', actor_loss.detach(), self.total_it)
            self.total_it += 1

            # update target networks
            if self.total_it % self.network_repl_freq == 0:
                self.update_target()

            return actor_loss.item(), critic_loss.item()

    def update_target(self):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

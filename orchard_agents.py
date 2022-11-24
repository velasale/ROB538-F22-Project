import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import namedtuple, deque
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import dqn
BATCH_SIZE = 50

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class AgentBase():
    def __init__(self) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 0
        # Class specific action
        self.action_type = 0
        # current position
        self.cur_pose = [0, 0]
        self.goal_pose = self.cur_pose
        # current comms channel (ROBOT ID)
        self.comms_channel = None
        self.policy_net = dqn.DQN(device).to(device)
        self.target_net = dqn.DQN(device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.eps = .3

    def reset_agent(self):
        # TODO: Used to reset the agent after each episode
        pass

    def optimize_agent(self):
        if len(self.policy_net.memory) < BATCH_SIZE:
            return None

        transitions = self.policy_net.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                             if s is not None])
        #state_batch = torch.cat(batch.state, 0)
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * .999) + reward_batch
        # print(expected_state_action_values.unsqueeze(1).size())
        # print(state_action_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_net.optimizer.step()

    def choose_move(self, action_areas, area_expectations, tstep):
        aa = np.ndarray.flatten(action_areas)
        state = np.concatenate((self.cur_pose, aa, area_expectations))
        state_t = torch.from_numpy(state).float()
        r = np.random.uniform(0, 1)
        if tstep % 1500 == 0 and self.eps != .05:
            self.eps -= .05
        if r < 1-self.eps:
            with torch.no_grad():
                logits = self.policy_net(state_t)
                #logits = torch.Tensor.cpu(logits)
                #softmax = nn.Softmax(dim=1)
                #pred_probab = softmax(logits)
                #idx = np.argmax(logits.data.numpy())
                idx = torch.Tensor.cpu(logits.max(0)[1])
                return action_areas[idx], idx
        else:
            rand_choice = np.random.randint(len(action_areas))
            return action_areas[rand_choice], rand_choice


class AgentPick(AgentBase):
    def __init__(self) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 100
        # Class specific action
        self.action_type = 1
        # current position
        self.cur_pose = [0, 0]
        self.goal_pose = self.cur_pose
        # current comms channel (ROBOT ID)
        self.comms_channel = None
        self.policy_net = dqn.DQN(device).to(device)
        self.target_net = dqn.DQN(device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.eps = .3


class AgentPrune(AgentBase):
    def __init__(self) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 200
        # Class specific action
        self.action_type = 2
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None

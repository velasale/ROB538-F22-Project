import pygame
import numpy as np
import time
import orchard_agents
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import namedtuple, deque
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt


BLACK = (0, 0, 0)  # BACKGROUND
WHITE = (255, 255, 255)  # BACKGROUND
TREEBASE = (0, 255, 0)  # NONE (GREEN)
TREE1 = (254, 138, 0)  # PICK (ORANGE)
TREE2 = (128, 75, 12)  # PRUNE (BROWN)
TREE3 = (255, 0, 157)  # PICK PRUNE (PINK)
TREE4 = (36, 187, 204)  # PRUNE PICK (BLUEISH)
RED = (255, 0, 0)  # RED (ACTION AREAS)
BLUE = (0, 0, 255)  # PICK ROBOT
PURPLE = (185, 57, 238)  # PRUNE ROBOT
YELLOW = (242, 238, 103)  # PRUNE ROBOT


class PygameRender():
    def __init__(self, orchard_map) -> None:
        self.map = orchard_map
        # x and y dimensions for map
        self.x_dim = np.shape(self.map.orchard_map)[0]
        self.y_dim = np.shape(self.map.orchard_map)[1]
        # width and margin for each cell
        self.margin = 2
        self.width = 20
        self.height = 20
        # size of our window based on the orchard map dimensions
        self.window_size = [(self.width * self.y_dim) + (self.y_dim * self.margin),
                            (self.height * self.x_dim) + (self.x_dim * self.margin)]
        # sprites loading for trees
        self.tree_apple_sprite = pygame.transform.scale(
            pygame.image.load("sprites/tree_apple.png"),
            (self.width, self.height))
        self.tree_prune_sprite = pygame.transform.scale(
            pygame.image.load("sprites/tree_prune.png"),
            (self.width, self.height))
        self.tree_apple_prune_sprite = pygame.transform.scale(
            pygame.image.load("sprites/tree_prune_apple.png"),
            (self.width, self.height))
        self.tree_prune_apple_sprite = pygame.transform.scale(
            pygame.image.load("sprites/tree_apple_prune.png"),
            (self.width, self.height))
        # start pygame
        pygame.init()
        # get screen
        self.screen = pygame.display.set_mode(self.window_size)
        # display
        pygame.display.set_caption("ORCHARD")

    def start(self, agents: list, max_ep: int, max_tstep: int):
        # tsteps and episodes
        tsteps = 0
        total_tsteps = 0
        eps = 0
        done = False
        reward_plot = []
        # sets clock
        clock = pygame.time.Clock()
        # loops forever until exit
        while not done:
            # 30fps
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # If user clicked close
                    done = True  # Flag that we are done so we exit this loop

            # main control loop for all agents
            states1 = []
            actions = []
            expectations = []
            for i in agents:
                # get valid moves for agent
                valid_moves, valid_keys = self.map.get_valid_moves(i.cur_pose, i.action_type)
                # if we have a valid move continue
                if len(valid_keys) > 0:
                    # get the surrounding area with sensors
                    # points, vals = self.map.get_surroundings(i.cur_pose, 3)
                    area, exp = self.map.get_area_expectations()
                    move, idx = i.choose_move(area, exp, total_tsteps)
                    state_t = torch.from_numpy(np.concatenate((i.cur_pose, np.ndarray.flatten(area), exp))).float()
                    states1.append(state_t)
                    actions.append(torch.from_numpy(np.array([idx])))
                    # update our map with our action choice
                    self.map.update_map(i.cur_pose, move, "interact", i.id)
                    # if we moved from a spot we need to update the agents internal current position
                    i.cur_pose = move

            for i in range(len(agents)):
                area, exp = self.map.get_area_expectations()
                new_state_t = torch.from_numpy(np.concatenate(
                    (agents[i].cur_pose, np.ndarray.flatten(area), exp))).float()
                reward = self.map.calculate_global_reward()
                agents[i].policy_net.memory.push(states1[i], actions[i], new_state_t,
                                                 torch.from_numpy(reward).float())
                agents[i].optimize_agent()
                if total_tsteps % 100:
                    agents[i].target_net.load_state_dict(agents[i].policy_net.state_dict())

                # draws everything
            self.draw_grid()
            # sleep to make it less fast, can take out if you want it sped up
            # time.sleep(.1)
            # updates display
            pygame.display.update()

            # if we are at max timestep increment episode and reset
            if tsteps >= max_tstep or self.map.check_complete():
                print("EPISODE : " + str(eps) + " COMPLETE")
                reward_plot.append(int(self.map.calculate_global_reward()))
                print(reward_plot[-1])
                # if we are at max episode then quit
                if eps >= max_ep:
                    plt.plot(np.arange(len(reward_plot)), reward_plot)
                    plt.show()
                    pygame.quit()
                    return
                # reset tsteps
                tsteps = 0
                # reset the agents and the map
                for i in agents:
                    i.reset_agent()
                self.map.reset_map(agents)
                eps += 1
            # increment timestep
            tsteps += 1
            total_tsteps += 1
        # quits
        pygame.quit()

    def draw_grid(self):
        # draw background
        self.screen.fill(BLACK)
        # Draw the grid from map
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                # If normal tree make square green
                if self.map.orchard_map[i][j] == -10:
                    color = TREEBASE
                # If action area make square green
                if self.map.orchard_map[i][j] == -20:
                    color = RED
                # If agent with ID in the picker robot range make square blue
                if self.map.orchard_map[i][j] >= 100 and self.map.orchard_map[i][j] < 200:
                    color = BLUE
                # If agent with ID in the pruner robot range make square blue
                if self.map.orchard_map[i][j] >= 200 and self.map.orchard_map[i][j] < 300:
                    color = PURPLE
                # If nothing in square make square blue
                if self.map.orchard_map[i][j] == 0:
                    color = WHITE
                # draw above
                pygame.draw.rect(self.screen,
                                 color,
                                 [(self.margin + self.width) * j + self.margin,
                                     (self.margin + self.height) *
                                     i + self.margin,
                                     self.width,
                                     self.height])
                # If tree is action sequence 1 use the sprite
                if self.map.orchard_map[i][j] == 1:
                    r = pygame.Rect((self.margin + self.width) * j + self.margin,
                                    (self.margin + self.height) *
                                    i + self.margin,
                                    self.width,
                                    self.height)
                    self.screen.blit(self.tree_apple_sprite, r)
                # If tree is action sequence 2 use the sprite
                if self.map.orchard_map[i][j] == 2:
                    r = pygame.Rect((self.margin + self.width) * j + self.margin,
                                    (self.margin + self.height) *
                                    i + self.margin,
                                    self.width,
                                    self.height)
                    self.screen.blit(self.tree_prune_sprite, r)
                # If tree is action sequence 3 use the sprite
                if self.map.orchard_map[i][j] == 3:
                    r = pygame.Rect((self.margin + self.width) * j + self.margin,
                                    (self.margin + self.height) *
                                    i + self.margin,
                                    self.width,
                                    self.height)
                    self.screen.blit(self.tree_apple_prune_sprite, r)
                # If tree is action sequence 4 use the sprite
                if self.map.orchard_map[i][j] == 4:
                    r = pygame.Rect((self.margin + self.width) * j + self.margin,
                                    (self.margin + self.height) *
                                    i + self.margin,
                                    self.width,
                                    self.height)
                    self.screen.blit(self.tree_prune_apple_sprite, r)

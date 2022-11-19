import pygame
import numpy as np
import time
import orchard_agents
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

        self.reward_flag = 0

    def start(self, agents: list, max_ep: int, max_tstep: int):
        #tsteps and episodes
        tsteps = 0
        eps = 0
        done = False
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
            for i in agents:

                # get valid moves for agent
                valid_moves, valid_keys = self.map.get_valid_moves(i.cur_pose, i.action_type)
                # if we have a valid move continue
                if len(valid_keys) > 0:
                    # get the surrounding area with sensors
                    points, vals = self.map.get_surroundings(i.cur_pose, 3)
                    # if internal channel is set we want to communicate
                    if i.comms_channel != None:
                        # finds the agent we want to communicate with
                        for j in agents:
                            if j.id == j.comms_channel:
                                # gets map from other agent
                                i.recieve_communication(j.send_communication())

                    # Agent chooses move doesnt do anything yet
                    # print("Valid moves and valid keys are: ", valid_moves, valid_keys)

                    # --- Step 1: Take next action ---
                    # a - Observe next state
                    move, key = i.choose_move(points, vals, valid_moves, valid_keys)

                    # b. Observe reward
                    reward = self.map.reward_map[move[0]][move[1]]

                    # --- Step 2: Choose A_prime from S_prime
                    valid_moves_prime, valid_keys_prime = self.map.get_valid_moves(move, i.action_type)
                    move_2, key_2 = i.choose_move(points, vals, valid_moves_prime, valid_keys_prime)

                    # --- Step 3: Update Q_sa_values of the agent
                    i.update_value(move, move_2, reward)
                    i.accumulated_reward = i.accumulated_reward + reward

                    if reward == 10:
                        # Tree found!
                        self.reward_flag = 1

                    # REMOVE RANDOM MOVE ONCE CHOOSE MOVE IMPLEMENTED ONLY FOR DEMO
                    # print(i.action_type)
                    # move, key = i.random_move(valid_moves, valid_keys)
                    # update our map with our action choice
                    self.map.update_map(i.cur_pose, move, key, i.id)
                    # if we moved from a spot we need to update the agents internal current position
                    if key != "interact":
                        i.cur_pose = move

                # Update epsilon
                i.epsilon = i.epsilon * 0.999
                # print(i.epsilon)

            # draws everything
            self.draw_grid()
            # sleep to make it less fast, can take out if you want it sped up
            # time.sleep(.1)
            # updates display
            pygame.display.update()

            # if we are at max timestep increment episode and reset
            if tsteps >= max_tstep or self.map.check_complete() or self.reward_flag == 1:

                self.reward_flag = 0

                print("EPISODE : " + str(eps) + " COMPLETE")
                # if we are at max episode then quit
                if eps >= max_ep:
                    pygame.quit()
                    return
                # reset tsteps
                tsteps = 0
                # reset the agents and the map
                for i in agents:
                    i.reset_agent()
                self.map.reset_map(agents)

                print(i.accumulated_reward)
                i.accumulated_reward = 0

                eps += 1
            # increment timestep
            tsteps += 1
        # quits
        pygame.quit()

        fig = plt.figure()
        plt.plot(rewards_evolution)
        fig = plt.figure()
        plt.imshow(i.q_sa_table)
        plt.colorbar()
        plt.show()

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

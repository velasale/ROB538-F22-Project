import pygame
import numpy as np
import time
import orchard_agents

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
                valid_moves, valid_keys, invalid_moves = self.map.get_valid_moves(i.cur_pose, i.action_type)
                if len(valid_keys) > 0:
                    # get agent position
                    start_pos = i.cur_pose.copy()
                    # get state for the given agent
                    if i.action_type == 2:
                        tree_state, count = self.map.get_prune_tree_state()
                    else:
                        tree_state, count = self.map.get_apple_tree_state()
                    # Choose a move
                    move, key, same_location = i.choose_move_tree_path(
                        tree_state, self.map.action_areas, valid_moves, valid_keys, start_pos, invalid_moves, count)
                    # If we did not move anywhere and stayed in the same location we give a large negative reward
                    if same_location:
                        reward = -50
                        other_state = None
                        if i.action_type == 2:
                            if reward > 0 and tree_finished == False:
                                other_state = self.map.get_apple_tree_state()
                            tree_state, count = self.map.get_prune_tree_state()
                        else:
                            if reward > 0 and tree_finished == False:
                                other_state = self.map.get_prune_tree_state()
                            tree_state, count = self.map.get_apple_tree_state()
                        i.update_next_state_path(tree_state, count)
                        i.update_buffer(reward)
                        i.policy.train()
                        if eps % 10 == 0:
                            i.update_epsilon()
                    # If we had a valid move
                    if move:
                        # update the map and get our reward
                        # CHANGE REWARD SCHEME HERE
                        # reward, tree_finished = self.map.update_map_local(
                        #     i.cur_pose, move, key, i.id, i.action_type, i.goal_position, i.goal_distance, i.start_position)
                        reward, tree_finished = self.map.update_map_diff(
                            i.cur_pose, move, key, i.id, i.action_type, i.goal_position, i.goal_distance, i.start_position)
                        # reward, tree_finished = self.map.update_map_global(
                        #     i.cur_pose, move, key, i.id, i.action_type, i.goal_position, i.goal_distance, i.start_position)

                        # if we interacted dont update pose
                        if key != "interact":
                            i.cur_pose = move
                        # if we recieved a reward
                        if reward:
                            # get updated states
                            other_state = None
                            if i.action_type == 2:
                                if reward > 0 and tree_finished == False:
                                    other_state = self.map.get_apple_tree_state()
                                tree_state, count = self.map.get_prune_tree_state()
                            else:
                                if reward > 0 and tree_finished == False:
                                    other_state = self.map.get_prune_tree_state()
                                tree_state, count = self.map.get_apple_tree_state()
                            #update and train
                            i.update_next_state_path(tree_state, count)
                            i.update_buffer(reward)
                            i.policy.train()
                            # update epsilon every 10 episodes
                            if eps % 10 == 0:
                                i.update_epsilon()

            self.draw_grid()
            # sleep to make it less fast, can take out if you want it sped up
            time.sleep(.1)
            # updates display
            pygame.display.update()

            # if we are at max timestep increment episode and reset
            if tsteps >= max_tstep or self.map.check_complete():
                print("EPISODE : " + str(eps) + " COMPLETE")
                print("EPSILON: ", agents[0].epsilon)
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
                eps += 1
            # increment timestep
            tsteps += 1
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

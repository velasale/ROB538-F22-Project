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
                # print(self.map.orchard_map)
                # if we have a valid move continue
                # print(valid_keys)
                if len(valid_keys) > 0:
                    # get the surrounding area with sensors
                    #points, vals = self.map.get_surroundings(i.cur_pose, 10)
                    # i.apply_sensor(points,vals,tsteps+1)
                    # if internal channel is set we want to communicate
                    # if i.comms_channel != None:
                    # finds the agent we want to communicate with
                    #    for j in self.agents:
                    #        if j.id == j.comms_channel:
                    # gets map from other agent
                    #            i.recieve_communication(j.send_communication())
                    # Agent chooses move doesnt do anything yet
                    start_pos = i.cur_pose.copy()
                    if i.action_type == 2:
                        tree_state, count = self.map.get_prune_tree_state()
                    else:
                        tree_state, count = self.map.get_apple_tree_state()
                    # move, key, actions = i.choose_move(points, vals, valid_moves, valid_keys, start_pos)
                    move, key, same_location = i.choose_move_tree_path(
                        tree_state, self.map.action_areas, valid_moves, valid_keys, start_pos, invalid_moves, count)
                    if i.action_type == 1:
                        print("CURRENT GOAL: ", i.goal_position)
                        print("CURRENT MOVE: ", move)
                        print("CURRENT MOVE KEY: ", key)
                        print("CURRENT PATH LIST: ", i.path_list)
#                    if i.action_type == 2:
                    #     print(i.goal_position)
                    #     print(i.goal_distance)
                    # # print(key)
                    if same_location:
                        reward = (-3-(i.goal_distance))
                        if i.action_type == 1:
                            print(reward)
                            print("")
                        other_state = None
                        if i.action_type == 2:
                            if reward > 0 and tree_finished == False:
                                other_state = self.map.get_apple_tree_state()
                            tree_state, count = self.map.get_prune_tree_state()
                        else:
                            if reward > 0 and tree_finished == False:
                                other_state = self.map.get_prune_tree_state()
                            tree_state, count = self.map.get_apple_tree_state()
                    # if other_state:
                    #    i.update_buffer_shared(actions, reward, other_state)
                    # else:
                        i.update_next_state_path(tree_state, count)
                        i.update_buffer(reward)
                        # i.policy.train_shared()
                        i.policy.train()

                    if move:
                       # REMOVE RANDOM MOVE ONCE CHOOSE MOVE IMPLEMENTED ONLY FOR DEMO
                        # move, key = i.random_move(valid_moves, valid_keys)
                        # update our map with our action choice
                        reward, tree_finished = self.map.update_map(
                            i.cur_pose, move, key, i.id, i.action_type, i.goal_position, i.goal_distance)
                        # if i.action_type == 2:
                        #     print(i.cur_pose)
                        # # # if eps % 100 == 0:
                        # print(self.map.orchard_map)
                        # print(actions)
                        # print(reward)
                        # print(i.cur_pose)
                        # if we moved from a spot we need to update the agents internal current position
                        if key != "interact":
                            i.cur_pose = move
                        # i.apply_sensor(next_points, next_vals,tsteps+1.5)
                        if reward:
                            if i.action_type == 1:
                                print(reward)
                                print("")
                            other_state = None
                            if i.action_type == 2:
                                if reward > 0 and tree_finished == False:
                                    other_state = self.map.get_apple_tree_state()
                                tree_state, count = self.map.get_prune_tree_state()
                            else:
                                if reward > 0 and tree_finished == False:
                                    other_state = self.map.get_prune_tree_state()
                                tree_state, count = self.map.get_apple_tree_state()
                        # if other_state:
                        #    i.update_buffer_shared(actions, reward, other_state)
                        # else:
                            i.update_next_state_path(tree_state, count)
                            i.update_buffer(reward)
                            # i.policy.train_shared()
                        i.policy.train()
                        # if we are at max timestep increment episode and reset
                    if eps % 10 == 0:
                        i.update_epsilon()
                    self.map.timestep += 1

            # draws everything
            self.draw_grid()
            # sleep to make it less fast, can take out if you want it sped up
            time.sleep(.2)
            # updates display
            pygame.display.update()

            # if we are at max timestep increment episode and reset
            if tsteps >= max_tstep or self.map.check_complete():
                print("EPISODE : " + str(eps) + " COMPLETE")
                print("EPSILON: ", agents[i].epsilon)
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

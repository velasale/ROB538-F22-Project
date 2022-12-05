import numpy as np
import pygame_render
import copy
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
# import orchard_agents


class OrchardMap():

    def __init__(self, row_height: int, row_description: list, top_buffer: int,
                 bottom_buffer: int, action_sequence: list,
                 action_map: list, tree_prob: list, tree_combos: list, seed: int) -> None:
        np.random.seed(seed)
        # how long are the rows
        self.row_height = row_height
        self.row_description = row_description
        # how much space on top of the orchard
        self.top_buffer = top_buffer
        # how much space on bottom of the orchard
        self.bottom_buffer = bottom_buffer
        # action sequences that are possible
        self.action_sequence = action_sequence
        # action mappings
        self.action_map = action_map
        # probability for each tree to have an action sequence
        self.tree_prob = tree_prob
        # tpyes of tree combos
        self.tree_combos = tree_combos
        # main orchard map and a copy of the original
        self.orchard_map = np.zeros((self.row_height + self.top_buffer + self.bottom_buffer, len(row_description)))
        self.checklist = None
        self.original_map = None
        self.action_areas = None
        self.picked_apples = 0
        self.pruned_trees = 0
        self.rewards = []
        self.pathfinding_map = None
        self.finder = AStarFinder()
        self.episode_rewards = []
        self.timestep = 0

    def create_map(self, agents: list = None) -> None:
        # Change every row except for buffer rows to the row_description
        for i in range(self.top_buffer, len(self.orchard_map) - self.bottom_buffer):
            for j in range(len(self.row_description)):
                # If there is a tree we assign a random weighted action sequence to that tree and put in the representation
                if self.row_description[j] == -10:
                    self.orchard_map[i][j] = np.random.choice(self.tree_combos, 1, p=self.tree_prob)
                # otherwise continue
                else:
                    self.orchard_map[i][j] = self.row_description[j]
        # Take a copy of the original map (used in update method)
        self.original_map = np.copy(self.orchard_map)
        self.checklist = self.create_checklist()
        self.action_areas = self.create_valid_action_areas()
        # Spawn the agents at the top center of the map LENGTH NEEDS TO BE LONGER THAN NUMBER OF AGENTS
        start = (len(self.row_description) // 2) - (len(agents) // 2)
        self.checklist = self.create_checklist()
        for i in range(len(agents)):
            self.orchard_map[0][start + i] = agents[i].robot_class
            # sets the start pose of agents and the ids
            agents[i].cur_pose = [0, start + i]
            agents[i].id = agents[i].robot_class + i

    def get_surroundings(self, start: list, sight_length: int):
        # Gets the sight_length x sight_length area around the agent
        left = max(start[1] - sight_length, 0)
        right = min(start[1] + sight_length, len(self.row_description)-1)
        down = min(start[0] + sight_length, self.row_height + self.top_buffer + self.bottom_buffer - 1)
        up = max(start[0] - sight_length, 0)
        points = []
        values = []
        # loop through and find the points and corresponding values for each cell around you
        for i in range(up, down+1):
            for j in range(left, right+1):
                points.append([i, j])
                values.append(self.orchard_map[i][j])
        return np.array(points), np.array(values)

    def get_appleness(self, start: list, sight_length: int):
        # Gets the sight_length x sight_length area around the agent
        left = max(start[1] - sight_length, 0)
        right = min(start[1] + sight_length, len(self.row_description)-1)
        down = min(start[0] + sight_length, self.row_height + self.top_buffer + self.bottom_buffer - 1)
        up = max(start[0] - sight_length, 0)
        points = []
        values = []
        apple_map = np.zeros([sight_length*2+1, sight_length*2+1])
        print('apple map')
        print(apple_map)
        # loop through and find the points and corresponding values for each cell around you
        for i in range(up, down+1):
            for j in range(left, right+1):
                # print(i,j)
                if self.orchard_map[i][j] == 3 or self.orchard_map[i][j] == 1:
                    apple_map[i-up, j-left] = 1
                else:
                    apple_map[i-up, j-left] = 0
        appleness = []
        for i in range(3):
            for j in range(3):
                temp = apple_map[2*i:2*i+3, 2*j:2*j+3].sum()
                if temp > 3:
                    appleness.append(2)
                elif temp >= 1:
                    appleness.append(1)
                else:
                    appleness.append(0)
        return appleness

    def get_valid_moves(self, start: list, action_type: int):
        # Finds all the valid moves for a given agent
        valid_moves = []
        valid_keys = []
        invalid_moves = []
        # Is down valid
        if start[0] < self.row_height + self.top_buffer + self.bottom_buffer - 1:
            down = self.orchard_map[start[0]+1, start[1]]
            if down == 0 or down == -20:
                valid_moves.append([start[0]+1, start[1]])
                valid_keys.append("down")
            else:
                invalid_moves.append([start[0]+1, start[1]])
        # Is up valid
        if start[0] > 0:
            up = self.orchard_map[start[0]-1, start[1]]
            if up == 0 or up == -20:
                valid_moves.append([start[0]-1, start[1]])
                valid_keys.append("up")
            else:
                invalid_moves.append([start[0]-1, start[1]])
        # Is right valid
        if start[1] < len(self.row_description)-1:
            right = self.orchard_map[start[0], start[1]+1]
            if right == 0 or right == -20:
                valid_moves.append([start[0], start[1]+1])
                valid_keys.append("right")
            # Checks if we can interact with a tree on our right (only works if our action type works for the action sequence)
            if right in self.action_map.keys() and action_type == self.action_map[right]:
                valid_moves.append([start[0], start[1]+1])
                valid_keys.append("interact")

            if right not in self.action_map.keys() and right != 0 and right != -20:
                invalid_moves.append([start[0], start[1]+1])
        # Is left valid
        if start[1] > 0:
            left = self.orchard_map[start[0], start[1]-1]
            if left == 0 or left == -20:
                valid_moves.append([start[0], start[1]-1])
                valid_keys.append("left")
            # Checks if we can interact with a tree on our right (only works if our action type works for the action sequence)
            if left in self.action_map.keys() and action_type == self.action_map[left]:
                valid_moves.append([start[0], start[1]-1])
                valid_keys.append("interact")
            if left not in self.action_map.keys() and left != 0 and left != -20:
                invalid_moves.append([start[0], start[1]-1])

        # returns list of x,y for all valid moves and a list of valid action keys: up, down, left, right, interact
        return valid_moves, valid_keys, invalid_moves

    def update_map_local(
            self, start: list, goal: list, key: str, agent_id: int, agent_type: int, path_goal, path_distance,
            start_position) -> None:

        if key == "interact":
            if agent_type == self.action_map[self.orchard_map[goal[0]][goal[1]]]:
                # Update tree
                self.orchard_map[goal[0]][goal[1]] = self.action_sequence[self.orchard_map[goal[0]][goal[1]]]
                tree_finished = True
                # Check if tree has been finished (NOT REALLY USED RIGHT NOW)
                if self.orchard_map[goal[0]][goal[1]] != -10:
                    tree_finished = False
                self.episode_rewards.append(1)
                # increase number picked
                if agent_type == 1:
                    self.picked_apples += 1
                elif agent_type == 2:
                    self.pruned_trees += 1
                # returns reward
                return (15 - (path_distance/4)), tree_finished
        else:
            # Update location
            self.orchard_map[start[0]][start[1]] = self.original_map[start[0]][start[1]]
            self.orchard_map[goal[0]][goal[1]] = agent_id
            self.episode_rewards.append(0)
            # checks to see if we could interact or not at the end of sequence, if not we get a negative reward
            moves, keys, invalid = self.get_valid_moves(goal, agent_type)
            if path_goal == goal and "interact" not in keys:
                # negative reward
                return (-3-path_distance/4), True
            else:
                return None, False

    def update_map_global(
            self, start: list, goal: list, key: str, agent_id: int, agent_type: int, path_goal, path_distance,
            start_position) -> None:
        if key == "interact":
            if agent_type == self.action_map[self.orchard_map[goal[0]][goal[1]]]:
                # update the tree
                self.orchard_map[goal[0]][goal[1]] = self.action_sequence[self.orchard_map[goal[0]][goal[1]]]
                tree_finished = True
                if self.orchard_map[goal[0]][goal[1]] != -10:
                    tree_finished = False
                # increment total
                self.episode_rewards.append(1)
                if agent_type == 1:
                    self.picked_apples += 1
                elif agent_type == 2:
                    self.pruned_trees += 1
                # Get total number of pick and prune for map
                combined = np.sum(self.original_map == 3) + np.sum(self.original_map == 4)
                n1 = np.sum(self.original_map == 1) + combined
                n2 = np.sum(self.original_map == 2) + combined
                combined_revealed = np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
                # G = %Complete + %Revealed - DistanceTravelled/4
                global_r2 = (((self.picked_apples+self.pruned_trees) / (n1+n2))
                             * 100) + ((((combined-combined_revealed)) / combined)*100) - (path_distance/4)
                return global_r2, tree_finished
        else:
            # if we move we change our previous location back to the original and update our id location
            self.orchard_map[start[0]][start[1]] = self.original_map[start[0]][start[1]]
            self.orchard_map[goal[0]][goal[1]] = agent_id
            self.episode_rewards.append(0)
            # check if we can interact at end of sequence if not then get reward
            moves, keys, invalid = self.get_valid_moves(goal, agent_type)
            if path_goal == goal and "interact" not in keys:
                combined = np.sum(self.original_map == 3) + np.sum(self.original_map == 4)
                # Get number of apples and prune
                n1 = np.sum(self.original_map == 1) + combined
                n2 = np.sum(self.original_map == 2) + combined
                # G = %Complete + %Revealed - DistanceTravelled/4
                combined_revealed = np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
                global_r2 = (((self.picked_apples+self.pruned_trees) / (n1+n2))
                             * 100) + ((((combined-combined_revealed)) / combined)*100) - (path_distance/4)
                return global_r2, True
            else:
                return None, False

    def update_map_diff(
            self, start: list, goal: list, key: str, agent_id: int, agent_type: int, path_goal, path_distance,
            start_position) -> None:

        if key == "interact":
            # Save number of apples picked and pruned before we update the map
            picked_prev = self.picked_apples
            pruned_prev = self.pruned_trees
            # save number of revealed before map update
            combined_revealed_prev = np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
            if agent_type == self.action_map[self.orchard_map[goal[0]][goal[1]]]:
                # update the tree
                self.orchard_map[goal[0]][goal[1]] = self.action_sequence[self.orchard_map[goal[0]][goal[1]]]
                tree_finished = True
                if self.orchard_map[goal[0]][goal[1]] != -10:
                    tree_finished = False
                # increment total
                self.episode_rewards.append(1)
                if agent_type == 1:
                    self.picked_apples += 1
                elif agent_type == 2:
                    self.pruned_trees += 1

                # Get total number of pick and prune for map
                combined = np.sum(self.original_map == 3) + np.sum(self.original_map == 4)
                n1 = np.sum(self.original_map == 1) + combined
                n2 = np.sum(self.original_map == 2) + combined
                combined_revealed = np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
                # G = %Complete + %Revealed - DistanceTravelled/4
                global_r2 = (((self.picked_apples+self.pruned_trees) / (n1+n2))
                             * 100) + ((((combined-combined_revealed)) / combined)*100) - (path_distance/4)

                # Counterfactual, get random goal location
                random_location = self.checklist[np.random.randint(len(self.checklist))]
                tree_t = self.orchard_map[random_location[0]][random_location[1]]
                # If we can interact at that location
                if tree_t in self.action_map.keys() and agent_type == self.action_map[tree_t]:
                    # Setup A* to find the total distance
                    self.pathfinding_map[random_location[0]][random_location[1]] = 1
                    pathfinding_grid = Grid(matrix=self.pathfinding_map)
                    start = pathfinding_grid.node(start_position[1], start_position[0])
                    end = pathfinding_grid.node(random_location[1], random_location[0])
                    path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
                    new_path_distance = len(path_list)
                    # If no path distance set it to 5
                    if new_path_distance == 0:
                        new_path_distance = 5
                    # Increase amount
                    pruned_prev += 1
                    # Increase number revealed if 3 or 4
                    if tree_t == 3 or tree_t == 4:
                        combined_revealed_prev += 1
                    self.pathfinding_map[random_location[0]][random_location[1]] = -1
                else:
                    # If no interaction calculate distance negative reward
                    self.pathfinding_map[random_location[0]][random_location[1]] = 1
                    pathfinding_grid = Grid(matrix=self.pathfinding_map)
                    start = pathfinding_grid.node(start_position[1], start_position[0])
                    end = pathfinding_grid.node(random_location[1], random_location[0])
                    path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
                    new_path_distance = len(path_list)
                    self.pathfinding_map[random_location[0]][random_location[1]] = -1

                # G_z-i = %Complete + %Revealed - TotalDistance (New action)
                global_dif = ((((picked_prev+pruned_prev)) / (n1+n2))
                              * 100) + ((((combined-combined_revealed_prev)) / combined) * 100) - (new_path_distance/4)
                # D
                diff_r = global_r2 - global_dif
                return diff_r, tree_finished
        else:
            # Save number of apples picked and pruned before we update the map
            picked_prev = self.picked_apples
            pruned_prev = self.pruned_trees
            # save number of revealed before map update
            combined_revealed_prev = np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
            # if we move we change our previous location back to the original and update our id location
            self.orchard_map[start[0]][start[1]] = self.original_map[start[0]][start[1]]
            self.orchard_map[goal[0]][goal[1]] = agent_id
            self.episode_rewards.append(0)
            # check if we can interact at end of sequence if not then get reward
            moves, keys, invalid = self.get_valid_moves(goal, agent_type)
            if path_goal == goal and "interact" not in keys:
                combined = np.sum(self.original_map == 3) + np.sum(self.original_map == 4)
                # Get number of apples and prune
                n1 = np.sum(self.original_map == 1) + combined
                n2 = np.sum(self.original_map == 2) + combined
                # G = %Complete + %Revealed - DistanceTravelled/4
                combined_revealed = np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
                global_r2 = (((self.picked_apples+self.pruned_trees) / (n1+n2))
                             * 100) + ((((combined-combined_revealed)) / combined)*100) - (path_distance/4)

                # Counterfactual, get random goal location
                random_location = self.checklist[np.random.randint(len(self.checklist))]
                tree_t = self.orchard_map[random_location[0]][random_location[1]]
                if tree_t in self.action_map.keys() and agent_type == self.action_map[tree_t]:
                    # Get distance
                    self.pathfinding_map[random_location[0]][random_location[1]] = 1
                    pathfinding_grid = Grid(matrix=self.pathfinding_map)
                    start = pathfinding_grid.node(start_position[1], start_position[0])
                    end = pathfinding_grid.node(random_location[1], random_location[0])
                    path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
                    new_path_distance = len(path_list)
                    # get new counts
                    if new_path_distance == 0:
                        new_path_distance = 5
                    pruned_prev += 1
                    if tree_t == 3 or tree_t == 4:
                        combined_revealed_prev += 1
                    self.pathfinding_map[random_location[0]][random_location[1]] = -1
                else:
                    # Get distance
                    self.pathfinding_map[random_location[0]][random_location[1]] = 1
                    pathfinding_grid = Grid(matrix=self.pathfinding_map)
                    start = pathfinding_grid.node(start_position[1], start_position[0])
                    end = pathfinding_grid.node(random_location[1], random_location[0])
                    path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
                    new_path_distance = len(path_list)
                    self.pathfinding_map[random_location[0]][random_location[1]] = -1

                # G_z-i = %Complete + %Revealed - TotalDistance (New action)
                global_dif = ((((picked_prev+pruned_prev)) / (n1+n2))
                              * 100) + ((((combined-combined_revealed_prev)) / combined) * 100) - (new_path_distance/4)
                # D
                diff_r = global_r2 - global_dif
                return diff_r, True
            else:
                return None, False

    def create_checklist(self):
        # creates a checklist containing all of the x,y location of trees to compare at the end of a timestep
        tree_checklist = []
        for i in range(np.shape(self.orchard_map)[0]):
            for j in range(len(self.row_description)):
                if self.orchard_map[i][j] in self.tree_combos:
                    tree_checklist.append([i, j])

        return np.array(tree_checklist)

    def create_valid_action_areas(self):
        # creates a checklist containing all of the x,y location of trees to compare at the end of a timestep
        tree_checklist = []
        for i in range(np.shape(self.orchard_map)[0]):
            for j in range(len(self.row_description)):
                if self.orchard_map[i][j] == -20:
                    tree_checklist.append([i, j])
        return np.array(tree_checklist)

    def check_complete(self):
        # Checks if all trees have had their action sequence completed
        for i in self.checklist:
            if self.orchard_map[i[0]][i[1]] != -10:
                return False
        return True

    def reset_map(self, agents: list):
        # resets the map back to original state
        print(f'picked {self.picked_apples} apples this episode')
        print(f'pruned {self.pruned_trees} trees this episode')
        picked = self.picked_apples
        pruned = self.pruned_trees
        self.orchard_map = np.copy(self.original_map)

        # self.rewards.append([self.picked_apples, self.pruned_trees])
        self.episode_rewards = []
        self.timestep = 0
        self.picked_apples = 0
        self.pruned_trees = 0

        # respawns the agents
        start = (len(self.row_description) // 2) - (len(agents) // 2)

        for i in range(len(agents)):
            self.orchard_map[0][start + i] = agents[i].robot_class
            # sets the start pose of agents and the ids
            agents[i].cur_pose = [0, start + i]
        # start2 = [np.random.randint(10),np.random.randint(5)]
        # self.orchard_map[start2[0]][start2[1]] = agents[0].robot_class
        # agents[0].cur_pose = [start2[0],start2[1]]
        combined = np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)

        n1 = np.sum(self.orchard_map == 1) + combined
        print(n1, 'total apples to pick this time')
        n2 = np.sum(self.orchard_map == 2) + combined
        print(n2, 'total trees to prune this time')
        apple_percent = (picked / n1)
        prune_percent = (pruned / n2)
        total = ((picked+pruned) / (n1+n2)) * 100
        print(total, 'Percent complete')
        self.rewards.append([total, apple_percent, prune_percent])

    def get_apple_tree_state(self):
        tree_state = []
        count = 0
        for i in self.checklist:
            if self.orchard_map[i[0]][i[1]] == 1 or self.orchard_map[i[0]][i[1]] == 3:
                tree_state.append(1)
                count += 1
            else:
                tree_state.append(0)
        return tree_state, count

    def get_prune_tree_state(self):
        tree_state = []
        count = 0
        for i in self.checklist:
            if self.orchard_map[i[0]][i[1]] == 2 or self.orchard_map[i[0]][i[1]] == 4:
                tree_state.append(1)
                count += 1
            else:
                tree_state.append(0)
        return tree_state, count


class OrchardSim():

    def __init__(self, orchard_map: OrchardMap, agents: list, tstep_max: int, ep_max: int) -> None:
        # driver
        self.map = orchard_map
        self.agents = agents
        self.tsep_max = tstep_max
        self.ep_max = ep_max
        self.render = None

    def run_gui(self):
        # runs gui
        self.render = pygame_render.PygameRender(self.map)
        self.render.start(self.agents, self.ep_max, self.tsep_max)

    def run(self):
        # UNTESTED
        tsteps = 0
        eps = 0
        while True:
            # main control loop for all agents
            for i in self.agents:
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
                    self.map.timestep += 1
                    if tsteps >= self.tsep_max or self.map.check_complete():
                        print("EPISODE : " + str(eps) + " COMPLETE")
                        # if we are at max episode then quit
                        if eps >= self.ep_max:
                            # VISUALIZE HERE
                            self.render = pygame_render.PygameRender(self.map)
                            self.render.start(self.agents, self.ep_max, self.tsep_max)
                            return
                        # reset tsteps
                        tsteps = 0
                        # reset the agents and the map
                        for i in self.agents:
                            i.reset_agent()
                        self.map.reset_map(self.agents)
                        eps += 1
                    # increment timestep
                    tsteps += 1

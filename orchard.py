import numpy as np
import pygame_render
import pickle as pkl
# import copy
# import orchard_agents
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


class OrchardMap():

    def __init__(self, row_height: int, row_description: list, top_buffer: int,
                 bottom_buffer: int, action_sequence: list,
                 action_map: list, tree_prob: list, tree_combos: list) -> None:
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
        self.cf_map = self.orchard_map.copy()
        self.checklist = self.create_checklist()
        self.original_map = None
        self.picked_apples = 0
        self.pruned_trees = 0
        self.total_apples = 0
        self.total_leaves = 0
        self.rewards = []
        self.episode_rewards = []
        self.timestep = 0
        self.global_reward = []
        self.episode_global_rewards = []
        self.num_trees = []
        self.travel_dists = []

    def create_map(self, agents: list = None) -> None:
        rng = np.random.default_rng(42)
        # rng = None
        # Change every row except for buffer rows to the row_description
        for i in range(self.top_buffer, len(self.orchard_map) - self.bottom_buffer):
            for j in range(len(self.row_description)):
                # If there is a tree we assign a random weighted action sequence to that tree and put in the representation
                if self.row_description[j] == -10:
                    if rng is None:
                        self.orchard_map[i][j] = np.random.choice(self.tree_combos, 1, p=self.tree_prob)
                    else:
                        self.orchard_map[i][j] = rng.choice(self.tree_combos, 1, p=self.tree_prob)
                # otherwise continue
                else:
                    self.orchard_map[i][j] = self.row_description[j]
        # Take a copy of the original map (used in update method)
        self.original_map = np.copy(self.orchard_map)
        # Spawn the agents at the top center of the map LENGTH NEEDS TO BE LONGER THAN NUMBER OF AGENTS
        start = (len(self.row_description) // 2) - (len(agents) // 2)
        self.checklist, other = self.create_checklist()
        points = np.random.choice(len(other), len(agents))
        for i in range(len(agents)):
            point_ind = points[i]
            self.orchard_map[other[point_ind][0]][other[point_ind][1]] = agents[i].robot_class + i
            # sets the start pose of agents and the ids
            agents[i].cur_pose = [other[point_ind][0],other[point_ind][1]]
            agents[i].id = agents[i].robot_class + i
        self.cf_map = self.orchard_map.copy()
        self.total_apples = np.sum(self.orchard_map == 1) + np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
        self.total_leaves = np.sum(self.orchard_map == 2) + np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
        self.num_trees.append([self.total_apples, self.total_leaves])
        temp = self.orchard_map.shape
        self.travel_dists = np.zeros([temp[0],temp[1],2])
        # print(self.num_trees[-1])
        # print(self.orchard_map)

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
        apple_map = np.zeros([sight_length*2+1,sight_length*2+1])
        # loop through and find the points and corresponding values for each cell around you
        for i in range(up, down+1):
            for j in range(left, right+1):
                # print(i,j)
                if self.orchard_map[i][j] == 3 or self.orchard_map[i][j] == 1:
                    apple_map[i-up,j-left] = 1
                else:
                    apple_map[i-up,j-left] = 0
        appleness = []
        for i in range(3):
            for j in range(3):
                temp = apple_map[2*i:2*i+3,2*j:2*j+3].sum()
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
        # Is down valid
        if start[0] < self.row_height + self.top_buffer + self.bottom_buffer - 1:
            down = self.orchard_map[start[0]+1, start[1]]
            if down == 0 or down == -20:
                valid_moves.append([start[0]+1, start[1]])
                valid_keys.append("down")
        # Is up valid
        if start[0] > 0:
            up = self.orchard_map[start[0]-1, start[1]]
            if up == 0 or up == -20:
                valid_moves.append([start[0]-1, start[1]])
                valid_keys.append("up")
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
        # returns list of x,y for all valid moves and a list of valid action keys: up, down, left, right, interact
        return valid_moves, valid_keys

    def update_map(self, start: list, goal: list, key: str, agent_id: int, agent_type: int) -> None:

        # here we can test different types of rewards
        
        # base idea is just "if you interact, get +10
        
        # counterfactual ideas
        # Pick a random action instead, our action - random action
        # 
        if key == "interact":
            # print('WE INTERACTED')
            # print('interacted at timestep', self.timestep, start, goal)
            # if we interact we update the action sequence to the next step of the goal area
            self.orchard_map[goal[0]][goal[1]] = self.action_sequence[self.orchard_map[goal[0]][goal[1]]]
            # print(self.orchard_map[goal[0]][goal[1]])
            self.episode_rewards.append(1)
            if agent_type == 1:
                self.picked_apples += 1
                tree_reward = self.get_pick_recursive(start)
            elif agent_type == 2:
                self.pruned_trees += 1
                tree_reward = self.get_prune_recursive(start)
            temp = self.orchard_map.shape
            self.travel_dists = np.zeros([temp[0],temp[1],2])
            return self.pruned_trees + self.picked_apples - tree_reward/10
        else:
            # if we move we change our previous location back to the original and update our id location
            self.orchard_map[start[0]][start[1]] = self.original_map[start[0]][start[1]]
            self.orchard_map[goal[0]][goal[1]] = agent_id
            self.episode_rewards.append(0)
            if agent_type == 1:
                tree_reward = self.get_pick_recursive(start)
            elif agent_type == 2:
                tree_reward = self.get_prune_recursive(start)
            temp = self.orchard_map.shape
            self.travel_dists = np.zeros([temp[0],temp[1],2])
            return self.pruned_trees + self.picked_apples - tree_reward/10
        
    def update_cf_map(self, start: list, goal: list, key: str, agent_id: int, agent_type: int) -> None:

        # here we can test different types of rewards
        # base idea is just "if you interact, get +10
        
        # counterfactual ideas
        # Pick a random action instead, our action - random action
        # 
        if agent_type == 1:
            tree_reward = self.get_pick_recursive(start)
        elif agent_type == 2:
            tree_reward = self.get_prune_recursive(start)
        temp = self.orchard_map.shape
        print('tree reward',tree_reward)
        self.travel_dists = np.zeros([temp[0],temp[1],2])
        if 'interact' in key:
            return self.pruned_trees+self.picked_apples - tree_reward/10 + 1
        else:
            return self.pruned_trees+self.picked_apples - tree_reward/10
        # if key == "interact":
        #     self.cf_map[goal[0]][goal[1]] = self.action_sequence[self.cf_map[goal[0]][goal[1]]]
        #     if agent_type == 1:
        #         temp = self.pruned_trees+(self.picked_apples+1)
        #     else:
        #         temp = (self.pruned_trees+1)+self.picked_apples - self.get_closest_tree(start)/10
        #     return temp
        # else:
        #     # if we move we change our previous location back to the original and update our id location
        #     self.cf_map[start[0]][start[1]] = self.original_map[start[0]][start[1]]
        #     self.cf_map[goal[0]][goal[1]] = agent_id
        #     return self.pruned_trees + self.picked_apples - self.get_closest_tree(goal)/10
        
    def create_checklist(self):
        # creates a checklist containing all of the x,y location of trees to compare at the end of a timestep
        tree_checklist = []
        space_checklist = []
        for i in range(np.shape(self.orchard_map)[0]):
            for j in range(len(self.row_description)):
                if self.orchard_map[i,j] != -20 and self.orchard_map[i,j] != 0:
                    tree_checklist.append([i, j])
                else:
                    space_checklist.append([i, j])
        tree_checklist.reverse()
        space_checklist.reverse()
        return np.array(tree_checklist), space_checklist

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
        self.orchard_map = np.zeros(self.original_map.shape)
        self.create_map(agents)
        self.rewards.append([self.picked_apples, self.pruned_trees])
        self.episode_global_rewards.append(self.global_reward)
        self.episode_rewards = []
        self.global_reward = []
        self.timestep=0
        self.picked_apples = 0
        self.pruned_trees = 0
        
        # # respawns the agents
        # start = (len(self.row_description) // 2) - (len(agents) // 2)

        # for i in range(len(agents)):
        #     self.orchard_map[0][start + i] = agents[i].robot_class
        #     # sets the start pose of agents and the ids
        #     agents[i].cur_pose = [0, start + i]
        # # start2 = [np.random.randint(10),np.random.randint(5)]
        # # self.orchard_map[start2[0]][start2[1]] = agents[0].robot_class
        # # agents[0].cur_pose = [start2[0],start2[1]]
        n1 = np.sum(self.orchard_map == 1) + np.sum(self.orchard_map == 3)
        print(self.total_apples, 'total apples to pick in the future')
        print(self.total_leaves, 'total leaves to prune in the future')
        self.cf_map = self.orchard_map.copy()

    def get_apple_tree_state(self, cf=False):
        tree_state = []
        if cf:
            for i in self.checklist:
                if self.cf_map[i[0]][i[1]] == 1 or self.cf_map[i[0]][i[1]] == 3:
                    tree_state.append(1)
                else:
                    tree_state.append(0)
        else:
            for i in self.checklist:
                if self.orchard_map[i[0]][i[1]] == 1 or self.orchard_map[i[0]][i[1]] == 3:
                    tree_state.append(1)
                else:
                    tree_state.append(0)
        return tree_state
    
    def get_prune_tree_state(self, cf=False):
        tree_state = []
        if cf:
            for i in self.checklist:
                if self.cf_map[i[0]][i[1]] == 2 or self.cf_map[i[0]][i[1]] == 4:
                    tree_state.append(1)
                else:
                    tree_state.append(0)
        else:
            for i in self.checklist:
                if self.orchard_map[i[0]][i[1]] == 2 or self.orchard_map[i[0]][i[1]] == 4:
                    tree_state.append(1)
                else:
                    tree_state.append(0)
        return tree_state

    def get_closest_tree(self, pose, cf=False):
        min_dist = 100
        if cf:
            for tree in self.checklist:
                if (0 <  self.cf_map[tree[0]][tree[1]]) and (self.cf_map[tree[0]][tree[1]]<= 4):
                    a = abs(tree[0] - pose[0]) + abs(tree[1] - pose[1]) 
                    min_dist = min(a,min_dist)
        else:
            for tree in self.checklist:
                if (0 <  self.orchard_map[tree[0]][tree[1]]) and (self.orchard_map[tree[0]][tree[1]]<= 4):
                    a = abs(tree[0] - pose[0]) + abs(tree[1] - pose[1]) 
                    min_dist = min(a,min_dist)
        return min_dist
    
    def get_closest_apple_tree(self, pose, cf=False):
        min_dist = 100
        if cf:
            for tree in self.checklist:
                if (1 == self.cf_map[tree[0]][tree[1]]) or (self.cf_map[tree[0]][tree[1]] == 3):
                    a = abs(tree[0] - pose[0]) + abs(tree[1] - pose[1]) 
                    min_dist = min(a,min_dist)
        else:
            for tree in self.checklist:
                if (1 == self.orchard_map[tree[0]][tree[1]]) or (self.orchard_map[tree[0]][tree[1]] == 3):
                    a = abs(tree[0] - pose[0]) + abs(tree[1] - pose[1]) 
                    min_dist = min(a,min_dist)
        return min_dist
    
    def get_closest_prune_tree(self, pose, cf=False):
        min_dist = 100
        if cf:
            for tree in self.checklist:
                if (2 ==  self.cf_map[tree[0]][tree[1]]) or (self.cf_map[tree[0]][tree[1]] == 4):
                    a = abs(tree[0] - pose[0]) + abs(tree[1] - pose[1]) 
                    min_dist = min(a,min_dist)
        else:
            for tree in self.checklist:
                if (2 ==  self.orchard_map[tree[0]][tree[1]]) or (self.orchard_map[tree[0]][tree[1]] == 4):
                    a = abs(tree[0] - pose[0]) + abs(tree[1] - pose[1]) 
                    min_dist = min(a,min_dist)
        return min_dist
    
    def get_prune_recursive(self,start):
        search_point = start.copy()
        self.travel_dists[search_point[0],search_point[1]] = 3
        if search_point[0] < self.row_height + self.top_buffer + self.bottom_buffer - 1:
            if self.travel_dists[search_point[0] + 1, search_point[1], 1] == 0:
                down = self.orchard_map[start[0]+1, start[1]]
                if down == 2 or down == 4:
                    self.travel_dists[search_point[0]+1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 1]
                elif down == 0 or -20:
                    self.travel_dists[search_point[0]+1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 0]
                else:
                    self.travel_dists[search_point[0]+1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 2]
        # Is up valid
        if search_point[0] > 0:
            if self.travel_dists[search_point[0] - 1, search_point[1], 1] == 0:
                up = self.orchard_map[start[0]-1, start[1]]
                if up == 2 or up == 4:
                    self.travel_dists[search_point[0]-1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 1]
                elif up == 0 or -20:
                    self.travel_dists[search_point[0]-1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 0]
                else:
                    self.travel_dists[search_point[0]-1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 2]
        # Is right valid
        if search_point[1] < len(self.row_description)-1:
            if self.travel_dists[search_point[0], search_point[1]+1, 1] == 0:
                right = self.orchard_map[start[0], start[1]+1]
                if right == 2 or right == 4:
                    self.travel_dists[search_point[0], search_point[1]+1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 1]
                    return int(self.travel_dists[search_point[0],search_point[1],0]+1)
                elif right == 0 or -20:
                    self.travel_dists[search_point[0], search_point[1]+1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 0]
                else:
                    self.travel_dists[search_point[0], search_point[1]+1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 2]
        # Is left valid
        if search_point[1] > 0:
            if self.travel_dists[search_point[0], search_point[1]-1, 1] == 0:
                left = self.orchard_map[start[0], start[1]-1]
                if left == 2 or left == 4:
                    self.travel_dists[search_point[0], search_point[1]-1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 1]
                    return int(self.travel_dists[search_point[0],search_point[1],0]+1)
                elif left == 0 or -20:
                    self.travel_dists[search_point[0], search_point[1]-1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 0]
                else:
                    self.travel_dists[search_point[0], search_point[1]-1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 2]
        down_dist, up_dist, left_dist, right_dist = 100,100,100,100
        if search_point[0] < self.row_height + self.top_buffer + self.bottom_buffer - 1:
            # print(search_point[0]+1, search_point[1])
            if self.travel_dists[search_point[0] + 1, search_point[1], 1] == 0:
                down_dist = self.get_prune_recursive([search_point[0]+1, search_point[1]])
        if start[0] > 0:
            # print(search_point[0]-1, search_point[1])
            if self.travel_dists[search_point[0] - 1, search_point[1], 1] == 0:
                up_dist = self.get_prune_recursive([search_point[0]-1, search_point[1]])
        if start[1] < len(self.row_description)-1:
            if self.travel_dists[search_point[0], search_point[1]+1, 1] == 0:
                left_dist = self.get_prune_recursive([search_point[0], search_point[1]+1])
        if start[1] > 0:
            if self.travel_dists[search_point[0], search_point[1]-1, 1] == 0:
                right_dist = self.get_prune_recursive([search_point[0], search_point[1]-1])
        temp = self.orchard_map.shape
        # print([down_dist,up_dist,left_dist,right_dist])
        return min([down_dist,up_dist,left_dist,right_dist])

    def get_pick_recursive(self,start):
        search_point = start.copy()
        self.travel_dists[search_point[0],search_point[1]] = 3

        if search_point[0] < self.row_height + self.top_buffer + self.bottom_buffer - 1:
            if self.travel_dists[search_point[0] + 1, search_point[1], 1] == 0:
                down = self.orchard_map[start[0]+1, start[1]]
                if down == 1 or down == 3:
                    self.travel_dists[search_point[0]+1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 1]
                elif down == 0 or -20:
                    self.travel_dists[search_point[0]+1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 0]
                else:
                    self.travel_dists[search_point[0]+1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 2]
        # Is up valid
        if start[0] > 0:
            if self.travel_dists[search_point[0] - 1, search_point[1], 1] == 0:
                up = self.orchard_map[start[0]-1, start[1]]
                if up == 1 or up == 3:
                    self.travel_dists[search_point[0]-1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 1]
                elif up == 0 or -20:
                    self.travel_dists[search_point[0]-1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 0]
                else:
                    self.travel_dists[search_point[0]-1, search_point[1],:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 2]
        # Is right valid
        if start[1] < len(self.row_description)-1:
            if self.travel_dists[search_point[0], search_point[1]+1, 1] == 0:
                right = self.orchard_map[start[0], start[1]+1]
                if right == 1 or right == 3:
                    self.travel_dists[search_point[0], search_point[1]+1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 1]
                    return int(self.travel_dists[search_point[0],search_point[1],0]+1)
                elif right == 0 or -20:
                    self.travel_dists[search_point[0], search_point[1]+1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 0]
                else:
                    self.travel_dists[search_point[0], search_point[1]+1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 2]
        # Is left valid
        if start[1] > 0:
            if self.travel_dists[search_point[0], search_point[1]-1, 1] == 0:
                left = self.orchard_map[start[0], start[1]-1]
                if left == 1 or left == 3:
                    self.travel_dists[search_point[0], search_point[1]-1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 1]
                    return int(self.travel_dists[search_point[0],search_point[1],0]+1)
                elif left == 0 or -20:
                    self.travel_dists[search_point[0], search_point[1]-1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 0]
                else:
                    self.travel_dists[search_point[0], search_point[1]-1,:] = [self.travel_dists[search_point[0],search_point[1],0]+1, 2]
        down_dist, up_dist, left_dist, right_dist = 100,100,100,100
        if search_point[0] < self.row_height + self.top_buffer + self.bottom_buffer - 1:
            if self.travel_dists[search_point[0] + 1, search_point[1], 1] == 0:
                down_dist = self.get_prune_recursive([search_point[0]+1, search_point[1]])
        if start[0] > 0:
            if self.travel_dists[search_point[0] - 1, search_point[1], 1] == 0:
                up_dist = self.get_prune_recursive([search_point[0]-1, search_point[1]])
        if start[1] < len(self.row_description)-1:
            if self.travel_dists[search_point[0], search_point[1]+1, 1] == 0:
                left_dist = self.get_prune_recursive([search_point[0], search_point[1]+1])
        if start[1] > 0:
            if self.travel_dists[search_point[0], search_point[1]-1, 1] == 0:
                right_dist = self.get_prune_recursive([search_point[0], search_point[1]-1])
        temp = self.orchard_map.shape
        # print([down_dist,up_dist,left_dist,right_dist])
        return min([down_dist,up_dist,left_dist,right_dist])
         
    def save_data(self,filepath=None):
        if filepath is None:
            filepath = 'GIVEFILEPATH'
        save_dict = {'Global Reward':self.episode_global_rewards, 'Pick/Prune':self.rewards, 'Num Trees': self.num_trees}
        with open(filepath + '.pkl', 'wb+') as file:
            pkl.dump(save_dict, file)

class DiscreteOrchardSim():

    def __init__(self, orchard_map: OrchardMap, agents: list, tstep_max: int, ep_max: int) -> None:
        # driver
        self.map = orchard_map
        self.agents = agents
        self.map.create_map(self.agents)
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
            # print(self.map.orchard_map)
            for i in self.agents:
                # get valid moves for agent
                valid_moves, valid_keys = self.map.get_valid_moves(i.cur_pose, i.action_type)
                
                # print(self.map.orchard_map)
                # if we have a valid move continue
                print(valid_keys)
                if len(valid_keys) > 0:
                    # get the surrounding area with sensors
                    # points, vals = self.map.get_surroundings(i.cur_pose, 3)
                    # alternatively, get the appleness
                    appleness = self.map.get_appleness(i.cur_pose, 3)
                    # if internal channel is set we want to communicate
                    if i.comms_channel != None:
                        # finds the agent we want to communicate with
                        for j in self.agents:
                            if j.id == j.comms_channel:
                                # gets map from other agent
                                i.recieve_communication(j.send_communication())
                    # Agent chooses move doesnt do anything yet
                    start_pos = i.cur_pose.copy()
                    # move, key = i.choose_move(points, vals, valid_moves, valid_keys, start_pos)
                    move, key = i.choose_move_apple(appleness, valid_moves, valid_keys)
                    # print(key)
                    # REMOVE RANDOM MOVE ONCE CHOOSE MOVE IMPLEMENTED ONLY FOR DEMO
                    # move, key = i.random_move(valid_moves, valid_keys)
                    # update our map with our action choice

                    reward = self.map.update_map(i.cur_pose, move, key, i.id)
                    # print(i.cur_pose)
                    # if we moved from a spot we need to update the agents internal current position
                    if key != "interact":
                        i.cur_pose = move
                    next_points, next_vals = self.map.get_surroundings(i.cur_pose, 3)
                    next_appleness = self.map.get_appleness(i.cur_pose, 3) 
                    # i.policy.train(points, vals, start_pos, key, reward, next_points, next_vals, i.cur_pose.copy())
                    i.policy.train_apple(appleness, key, reward, next_appleness)
                    # if we are at max timestep increment episode and reset
                    if tsteps % 10 == 0:                        
                        i.update_epsilon()
                    
                    if tsteps >= self.tsep_max or self.map.check_complete():
                        print("EPISODE : " + str(eps) + " COMPLETE")
                        print(f'{self.agents[0].missed_interacts} missed apple picks')
                        print(f'{self.agents[1].missed_interacts} missed prunes')
                        self.agents[0].missed_interacts = 0
                        self.agents[1].missed_interacts = 0
                        # if we are at max episode then quit
                        if eps >= self.ep_max:
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


class OrchardSim():

    def __init__(self, orchard_map: OrchardMap, agents: list, tstep_max: int, ep_max: int) -> None:
        # driver
        self.map = orchard_map
        self.agents = agents

        self.map.create_map(self.agents)
        self.tsep_max = tstep_max
        self.ep_max = ep_max
        self.render = None
        self.missed = []


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
            # print(self.map.orchard_map)
            for count, i in enumerate(self.agents):
                c = 2 % (count+1)
                # get valid moves for agent
                # print('cur pose', i.cur_pose)
                valid_moves, valid_keys = self.map.get_valid_moves(i.cur_pose, i.action_type)
                # print(self.map.orchard_map)
                # if we have a valid move continue
                # print(valid_keys)
                if len(valid_keys) > 0:
                    # get the surrounding area with sensors
                    points, vals = self.map.get_surroundings(i.cur_pose, 10)
                    # i.apply_sensor(points,vals,tsteps+1)
                    # if internal channel is set we want to communicate
                    if i.comms_channel != None:
                        # finds the agent we want to communicate with
                        for j in self.agents:
                            if j.id == j.comms_channel:
                                # gets map from other agent
                                i.recieve_communication(j.send_communication())
                    # Agent chooses move doesnt do anything yet
                    start_pos = i.cur_pose.copy()
                    if i.action_type == 2:
                        tree_state = self.map.get_prune_tree_state()
                    else:
                        tree_state = self.map.get_apple_tree_state()
                    # move, key, actions = i.choose_move(points, vals, valid_moves, valid_keys, start_pos)
                    move, key, actions, missed = i.choose_move_tree(tree_state, valid_moves, valid_keys, start_pos, self.agents[c].cur_pose)
                    cf_move, cf_key, cf_actions = i.random_move(valid_moves, valid_keys)
                    # print(key)

                    # REMOVE RANDOM MOVE ONCE CHOOSE MOVE IMPLEMENTED ONLY FOR DEMO
                    # move, key = i.random_move(valid_moves, valid_keys)
                    # update our map with our action choice
                    # print(self.map.orchard_map)
                    # print(actions)
                    # print(self.map.cf_map)
                    # print(cf_actions)
                    cf_reward = self.map.update_cf_map(i.cur_pose, cf_move, valid_keys, i.id, i.action_type)
                    reward = self.map.update_map(i.cur_pose, move, key, i.id, i.action_type)
                    
                    # if missed:
                    #     print(f'reward v cf reward, {reward} v {cf_reward}')
                    # if we moved from a spot we need to update the agents internal current position
                    if key != "interact":
                        i.cur_pose = move
                    if cf_key != "interact":
                        i.cur_cf_pose = cf_move
                    else:
                        i.cur_cf_pose = i.cur_pose.copy()
                    next_points, next_vals = self.map.get_surroundings(i.cur_pose, 3)
                    # i.apply_sensor(next_points, next_vals,tsteps+1.5)
                    if i.action_type == 2:
                        tree_state = self.map.get_prune_tree_state()
                        cf_state = self.map.get_prune_tree_state(True)
                        # if len
                    else:
                        tree_state = self.map.get_apple_tree_state()
                        cf_state = self.map.get_apple_tree_state(True)
                        
                    i.update_next_state(tree_state, i.cur_pose.copy(), self.agents[c].cur_pose)
                    i.update_cf_state(cf_state, i.cur_cf_pose.copy(),self.agents[c].cur_pose)
                    i.update_buffer(actions, reward, cf_reward)
                    i.policy.train()
                    # if we are at max timestep increment episode and reset
                    if tsteps % 20 ==0:
                        i.update_epsilon()
                    self.map.timestep += 1
                    i.cur_cf_pose = i.cur_pose.copy()
                    self.map.cf_map = self.map.orchard_map.copy()
                    self.map.global_reward.append(100*self.map.picked_apples*self.map.pruned_trees/self.map.total_apples/self.map.total_leaves)
                    if tsteps >= self.tsep_max or self.map.check_complete():
                        print("EPISODE : " + str(eps) + " COMPLETE")
                        # if we are at max episode then quit
                        print(f'{self.agents[0].missed_interacts} missed apple picks')
                        print(f'{self.agents[1].missed_interacts} missed prunes')
                        print(self.agents[0].epsilon, self.agents[1].epsilon)
                        self.missed.append([self.agents[0].missed_interacts,self.agents[1].missed_interacts])
                        self.agents[0].missed_interacts = 0
                        self.agents[1].missed_interacts = 0
                        if eps >= self.ep_max:
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
            if eps % 100 == 0:
                print(self.map.orchard_map)
                print(actions)
                print(reward)

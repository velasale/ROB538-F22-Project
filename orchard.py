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

    def create_map(self, agents: list = None) -> None:
        rng = np.random.default_rng(42)
        # Change every row except for buffer rows to the row_description
        for i in range(self.top_buffer, len(self.orchard_map) - self.bottom_buffer):
            for j in range(len(self.row_description)):
                # If there is a tree we assign a random weighted action sequence to that tree and put in the representation
                if self.row_description[j] == -10:
                    self.orchard_map[i][j] = rng.choice(self.tree_combos, 1, p=self.tree_prob)
                # otherwise continue
                else:
                    self.orchard_map[i][j] = self.row_description[j]
        # Take a copy of the original map (used in update method)
        self.original_map = np.copy(self.orchard_map)
        # Spawn the agents at the top center of the map LENGTH NEEDS TO BE LONGER THAN NUMBER OF AGENTS
        start = (len(self.row_description) // 2) - (len(agents) // 2)
        self.checklist = self.create_checklist()
        for i in range(len(agents)):
            self.orchard_map[0][start + i] = agents[i].robot_class
            # sets the start pose of agents and the ids
            agents[i].cur_pose = [0, start + i]
            agents[i].id = agents[i].robot_class + i
        self.cf_map = self.orchard_map.copy()
        self.total_apples = np.sum(self.orchard_map == 1) + np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
        self.total_leaves = np.sum(self.orchard_map == 2) + np.sum(self.orchard_map == 3) + np.sum(self.orchard_map == 4)
        self.num_trees.append([self.total_apples, self.total_leaves])
        print(self.num_trees[-1])

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
        print('apple map')
        print(apple_map)
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
            # print(self.orchard_map[goal[0]][goal[1]])
            # print(self.action_sequence[self.orchard_map[goal[0]][goal[1]]])
            self.orchard_map[goal[0]][goal[1]] = self.action_sequence[self.orchard_map[goal[0]][goal[1]]]
            # print(self.orchard_map[goal[0]][goal[1]])
            self.episode_rewards.append(1)
            if agent_type == 1:
                self.picked_apples += 1
            elif agent_type == 2:
                self.pruned_trees += 1
            return 10
        else:
            # if we move we change our previous location back to the original and update our id location
            self.orchard_map[start[0]][start[1]] = self.original_map[start[0]][start[1]]
            self.orchard_map[goal[0]][goal[1]] = agent_id
            self.episode_rewards.append(0)
            return -1
        
    def update_cf_map(self, start: list, goal: list, key: str, agent_id: int, agent_type: int) -> None:

        # here we can test different types of rewards
        
        # base idea is just "if you interact, get +10
        
        # counterfactual ideas
        # Pick a random action instead, our action - random action
        # 
        if key == "interact":
            self.cf_map[goal[0]][goal[1]] = self.action_sequence[self.cf_map[goal[0]][goal[1]]]
            return 10
        else:
            # if we move we change our previous location back to the original and update our id location
            self.cf_map[start[0]][start[1]] = self.original_map[start[0]][start[1]]
            self.cf_map[goal[0]][goal[1]] = agent_id
            return -1
        
    def create_checklist(self):
        # creates a checklist containing all of the x,y location of trees to compare at the end of a timestep
        tree_checklist = []
        for i in range(np.shape(self.orchard_map)[0]):
            for j in range(len(self.row_description)):
                if self.orchard_map[i,j] != -20 and self.orchard_map[i,j] != 0:
                    tree_checklist.append([i, j])
        tree_checklist.reverse()
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
        print(n1, 'total apples to pick in the future')
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
                    i.update_epsilon()
                    
                    if tsteps >= self.tsep_max or self.map.check_complete():
                        print("EPISODE : " + str(eps) + " COMPLETE")
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
                    move, key, actions = i.choose_move_tree(tree_state, valid_moves, valid_keys, start_pos)
                    cf_move, cf_key, cf_actions = i.random_move(valid_moves, valid_keys)
                    # print(key)

                    # REMOVE RANDOM MOVE ONCE CHOOSE MOVE IMPLEMENTED ONLY FOR DEMO
                    # move, key = i.random_move(valid_moves, valid_keys)
                    # update our map with our action choice
                    # print(self.map.orchard_map)
                    # print(actions)
                    # print(self.map.cf_map)
                    # print(cf_actions)
                    cf_reward = self.map.update_cf_map(i.cur_pose, cf_move, cf_key, i.id, i.action_type)
                    reward = self.map.update_map(i.cur_pose, move, key, i.id, i.action_type)
                    
                    # if eps % 100 == 0:
                    #     print(self.map.orchard_map)
                    #     print(actions)
                    #     print(reward)
                    # print(i.cur_pose)
                    # if we moved from a spot we need to update the agents internal current position
                    if key != "interact":
                        i.cur_pose = move
                    if cf_key != "interact":
                        i.cur_cf_pose = cf_move
                    next_points, next_vals = self.map.get_surroundings(i.cur_pose, 3)
                    # i.apply_sensor(next_points, next_vals,tsteps+1.5)
                    if i.action_type == 2:
                        tree_state = self.map.get_prune_tree_state()
                        cf_state = self.map.get_prune_tree_state(True)
                        # if len
                    else:
                        tree_state = self.map.get_apple_tree_state()
                        cf_state = self.map.get_apple_tree_state(True)
                        
                    i.update_next_state(tree_state, i.cur_pose.copy())
                    i.update_cf_state(cf_state, i.cur_cf_pose.copy())
                    i.update_buffer(actions, reward, cf_reward)
                    i.policy.train()
                    # if we are at max timestep increment episode and reset
                    i.update_epsilon()
                    self.map.timestep += 1
                    i.cur_cf_pose = i.cur_pose.copy()
                    self.map.cf_map = self.map.orchard_map.copy()
                    self.map.global_reward.append(100*self.map.picked_apples*self.map.pruned_trees/self.map.total_apples/self.map.total_leaves)
                    if tsteps >= self.tsep_max or self.map.check_complete():
                        print("EPISODE : " + str(eps) + " COMPLETE")
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

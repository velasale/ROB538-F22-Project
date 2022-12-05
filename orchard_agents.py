import numpy as np
from learning import DiscreteLearning, LearningBaseClass, SAC, SACLimited
import copy
import torch
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
# memory key
# 0 = empty
# 1 = us
# 2 = empty tree
# 3 = tree with apple
# 4 = tree with branch
# 5 = tree with branch and apple
# 6 = other robot
# 7 = unseen


class AgentBase():
    def __init__(self, field_size=[10, 5]) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 0
        # Class specific action
        self.action_type = 0
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None
        # memory bank the size of the field.
        self.field_size = field_size
        self.memory = np.zeros([field_size[0], field_size[1], 5, 2])
        # intializing 0,0,0 to be our location
        self.memory[0, 0, 0, 0] = 1
        # saving a copy of our state so we can implement state/next_state
        self.prev_state = self.memory.copy()
        # initialize the learner
        self.policy = LearningBaseClass()
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 0.9

    def reset_agent(self):
        # TODO: Used to reset the agent after each episode
        self.memory = np.zeros([self.field_size[0], self.field_size[1], 5, 2])
        # intializing 0,0,0 to be our location
        self.memory[0, 0, 0, 0] = 1
        self.prev_state = self.memory.copy()

    def random_move(self, valid_moves, valid_keys):
        # randomly chooses a valid move, takes in a list of valid x,y moves and the corresponding valid keys
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        choice = np.random.randint(len(valid_moves))
        return valid_moves[choice], valid_keys[choice]

    def send_communication(self):
        # TODO: Should return internal map
        return self.memory

    def recieve_communication(self, comms):
        # assuming comms is the map from the other agent
        for i in range(self.field_size[0]):
            for j in range(self.field_size[1]):
                if self.memory[i, j, 1] > comms[i, j, 1]:
                    self.memory[i, j, :] = comms[i, j, :]
        self.comms_channel = None
        pass

    def request_communication(self, id):
        # TODO: Sets internal comms channel to the id of the robot we want a map from. This is then checked and sent over
        # in orchardsim.
        # SHOULD BE SET INTERNALLY
        self.comms_channel = id

    def choose_move(self, observed_points, observed_vals, valid_moves, valid_keys, cur_pos=None):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        if np.random.rand() > self.epsilon:
            action, action_key = self.policy.select_action(
                observed_points, observed_vals, valid_moves, valid_keys, cur_pos)
        else:
            action, action_key = self.random_move(valid_moves, valid_keys)

        return action, action_key

    def apply_sensor(self, sensor_location, sensor_reading, timestep):
        # here we assume that the sensor reading is a square,
        # smallest corner at sensor start
        # need to fix this so that it turns the state info into our info and saves last state in prev_state
        self.prev_state = self.memory.copy()
        sensor_size = len(sensor_reading)
        temp_reading = []
        for read, loc in zip(sensor_reading, sensor_location):
            if read == -10:
                # empty tree
                temp_reading = [0, 0, 1, 1, 0]
            elif read == 1:
                # tree with apple
                temp_reading = [1, 0, 1, 1, 0]
            elif read == 2:
                # tree with prune
                temp_reading = [0, 1, 1, 1, 0]
            elif read == 4:
                # tree with prune and apple
                temp_reading = [1, 1, 1, 1, 0]
            elif read >= 100:
                # robot
                temp_reading = [0, 0, 1, 0, 1]
            else:
                temp_reading = [0, 0, 1, 0, 0]
            self.memory[loc[0], loc[1], :, 1] = timestep
            self.memory[loc[0], loc[1], :, 0] = temp_reading

    def update_epsilon(self):
        self.epsilon *= 0.999

    def encode_memory(self):
        # yes this is slow, ill optimize later if needed
        one_hot_memory = torch.zeros(
            [self.field_size[0], self.field_size[1], 8])
        for i, row in enumerate(self.memory):
            for j, point in enumerate(row):
                one_hot_memory[i, j, point] = 1
        return one_hot_memory

    def update_buffer(self, actions, reward):
        self.policy.update_buffer(
            self.prev_state[:, :, :, 0], actions, reward, self.memory[:, :, :, 0])


class AgentPickDiscrete(AgentBase):
    def __init__(self, field_size=[10, 5]) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 100
        # Class specific action
        self.action_type = 1
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None
        # memory bank the size of the field.
        self.memory = np.zeros([field_size[0], field_size[1], 2])
        # intializing 0,0,0 to be our location
        self.memory[0, 0, 0] = 1
        # initialize the learner
        self.policy = DiscreteLearning(field_size)
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 0.9

    def apply_sensor(self, sensor_location, sensor_reading, timestep):
        pass

    def choose_move_apple(self, appleness, valid_moves, valid_keys):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        if np.random.rand() > self.epsilon:
            action, action_key = self.policy.select_action_apple(
                appleness, valid_moves, valid_keys)
        else:
            action, action_key = self.random_move(valid_moves, valid_keys)

        return action, action_key


class AgentPruneDiscrete(AgentBase):
    def __init__(self, field_size=[10, 5]) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 200
        # Class specific action
        self.action_type = 2
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None
        # memory bank the size of the field.
        self.memory = np.zeros([field_size[0], field_size[1], 2])
        # intializing 0,0,0 to be our location
        self.memory[0, 0, 0] = 1

        # initialize the learner
        self.policy = DiscreteLearning(field_size)
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 0.9


class AgentPickSAC(AgentBase):
    def __init__(self, field_size=[10, 5]) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 100
        # Class specific action
        self.action_type = 1
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None
        # memory bank the size of the field.
        self.memory = np.zeros([field_size[0], field_size[1], 5, 2])

        self.field_size = field_size
        # intializing 0,0,0 to be our location
        self.memory[0, 0, 0, 0] = 1

        self.prev_state = self.memory.copy()
        # initialize the learner
        self.policy = SAC(field_size[0]*field_size[1]*5, 5)
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 0.9
        self.action_order = ['left', 'right', 'up', 'down', 'interact']

    def choose_move(self, observed_points, observed_vals, valid_moves, valid_keys, cur_pos=None):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])
        if np.random.rand() > self.epsilon:
            action, action_key, actions = self.policy.select_action(
                self.memory[:, :, :, 0], valid_moves, valid_keys)
            actions = actions.detach().tolist()
        else:
            action, action_key, actions = self.random_move(
                valid_moves, valid_keys)

        return action, action_key, actions

    def random_move(self, valid_moves, valid_keys):
        # randomly chooses a valid move, takes in a list of valid x,y moves and the corresponding valid keys
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        choice = np.random.randint(len(valid_moves))
        a = self.action_order.index(valid_keys[choice])
        act = [0]*5
        act[a] = 1
        return valid_moves[choice], valid_keys[choice], act


class AgentPruneSAC(AgentBase):
    def __init__(self, field_size=[10, 5]) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 200
        # Class specific action
        self.action_type = 2
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None
        # memory bank the size of the field.
        self.memory = np.zeros([field_size[0], field_size[1], 5, 2])

        self.field_size = field_size
        # intializing 0,0,0 to be our location
        self.memory[0, 0, 0, 0] = 1
        # initialize the learner
        self.policy = SAC(field_size[0]*field_size[1]*5, 5, 'prune_agent')

        self.prev_state = self.memory.copy()
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 1
        self.action_order = ['left', 'right', 'up', 'down', 'interact']

    def choose_move(self, observed_points, observed_vals, valid_moves, valid_keys, cur_pos=None):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])
        if np.random.rand() > self.epsilon:
            action, action_key, actions = self.policy.select_action(
                self.memory[:, :, :, 0], valid_moves, valid_keys)
            actions = actions.detach().tolist()
        else:
            action, action_key, actions = self.random_move(
                valid_moves, valid_keys)

        return action, action_key, actions

    def random_move(self, valid_moves, valid_keys):
        # randomly chooses a valid move, takes in a list of valid x,y moves and the corresponding valid keys
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        choice = np.random.randint(len(valid_moves))
        a = self.action_order.index(valid_keys[choice])
        act = [0]*5
        act[a] = 1
        return valid_moves[choice], valid_keys[choice], act


class AgentPickSAClimited(AgentBase):
    def __init__(self, num_trees, opposite_buffer, shared_buffer, action_dim) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 100
        # Class specific action
        self.action_type = 1
        # current position
        self.cur_pose = [0, 0]
        # pathfinding
        self.action_dim = action_dim
        self.pathfinding_map = None
        self.finder = AStarFinder()
        self.goal_position = self.cur_pose
        self.start_position = self.cur_pose
        self.path_list = []
        self.goal_distance = 0
        # memory bank the size of the field.
        # initialize the learner
        self.policy = SACLimited(2+num_trees, action_dim,
                                 opposite_buffer, shared_buffer, "pick_agent")
        # epsilon for exploration
        self.epsilon = 0.9
        self.idx = 0
        self.state = []
        self.act = []
        self.prev_state = []
        self.action_order = ['left', 'right', 'up', 'down', 'interact']

    def choose_move_tree(self, tree_states, valid_moves, valid_keys, cur_pos):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])

        self.state = tree_states + cur_pos

        if np.random.rand() > self.epsilon:
            action, action_key, actions = self.policy.select_action(
                tree_states, valid_moves, valid_keys, cur_pos)
            actions = actions.detach().tolist()
        else:
            action, action_key, actions = self.random_move(
                valid_moves, valid_keys)

        return action, action_key, actions

    def choose_move_tree_path(
            self, tree_states, valid_action_areas, valid_moves, valid_keys, cur_pos, invalid_moves, count):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])
        temp_map = np.copy(self.pathfinding_map)
        for i, p in enumerate(invalid_moves):
            temp_map[p[0]][p[1]] = -1
        pathfinding_grid = Grid(matrix=temp_map)

        if len(self.path_list) == 0:
            if "interact" in valid_keys:
                return valid_moves[valid_keys.index("interact")], "interact", False
            self.act = [0]*self.action_dim
            self.start_position = self.cur_pose
            #self.state = tree_states + list(np.ndarray.flatten(np.array(valid_action_areas))) + cur_pos
            self.state = [count] + tree_states + cur_pos
            self.start_position = cur_pos
            if np.random.rand() > self.epsilon:
                idx = self.policy.select_action_path(self.state)
                self.idx = idx
                self.new_goal = True
                self.goal_position = [valid_action_areas[idx][0], valid_action_areas[idx][1]]
                self.act[idx] = 1
            else:
                idx = np.random.randint(len(valid_action_areas))
                self.idx = idx
                self.new_goal = True
                self.goal_position = [valid_action_areas[idx][0], valid_action_areas[idx][1]]
                self.act[idx] = 1

        start = pathfinding_grid.node(self.cur_pose[1], self.cur_pose[0])
        end = pathfinding_grid.node(valid_action_areas[self.idx][1], valid_action_areas[self.idx][0])
        self.path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
        if len(self.path_list) == 1:
            # print("HERLKJERLKEJRLKEJRLKEJR")
            self.path_list = []
            return None, None, True
        if len(self.path_list) == 0:
            # print("HERLKJERLKEJRLKEJRLKEJR")
            self.path_list = []
            return None, None, False

        self.path_list.pop(0)
        if self.new_goal:
            self.new_goal = False
            self.goal_distance = len(self.path_list)

        next_move = self.path_list.pop(0)
        action = None
        action_key = None

        for i, p in enumerate(valid_moves):
            if next_move[1] == p[0] and next_move[0] == p[1]:
                action = p
                action_key = valid_keys[i]
        if action_key == None:
            self.path_list = []
            return None, None, False

        return action, action_key, False

    # def choose_move_tree_path(self, tree_states, valid_action_areas, valid_moves, valid_keys, cur_pos, invalid_moves):
    #     # randomly chooses a valid move
    #     # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
    #     # returns the [x,y] of next move and the key: up, down, left, right or interact
    #     # print(self.memory[:,:,2,0])

    #     if len(self.path_list) == 0:
    #         self.act = [0]*self.action_dim
    #         if "interact" in valid_keys:
    #             return valid_moves[valid_keys.index("interact")], "interact"
    #         self.start_position = self.cur_pose
    #         temp_map = np.copy(self.pathfinding_map)
    #         for i, p in enumerate(invalid_moves):
    #             temp_map[p[0]][p[1]] = -1
    #             # print(temp_map)
    #         pathfinding_grid = Grid(matrix=temp_map)

    #         self.state = tree_states + cur_pos
    #         if np.random.rand() > self.epsilon:
    #             idx = self.policy.select_action_path(self.state)
    #             self.idx = idx
    #             self.goal_position = [valid_action_areas[idx][0], valid_action_areas[idx][1]]
    #             start = pathfinding_grid.node(self.cur_pose[1], self.cur_pose[0])
    #             end = pathfinding_grid.node(valid_action_areas[idx][1], valid_action_areas[idx][0])
    #             self.path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
    #             if len(self.path_list) <= 1:
    #                 return None, None
    #             self.path_list.pop(0)
    #             self.goal_distance = len(self.path_list)
    #             self.act[idx] = 1
    #         else:
    #             idx = np.random.randint(len(valid_action_areas))
    #             self.idx = idx
    #             self.goal_position = [valid_action_areas[idx][0], valid_action_areas[idx][1]]
    #             start = pathfinding_grid.node(self.cur_pose[1], self.cur_pose[0])
    #             end = pathfinding_grid.node(valid_action_areas[idx][1], valid_action_areas[idx][0])
    #             self.path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
    #             if len(self.path_list) <= 1:
    #                 return None, None
    #             self.path_list.pop(0)
    #             self.goal_distance = len(self.path_list)
    #             self.act[idx] = 1
    #     next_move = self.path_list.pop(0)
    #     action = None
    #     action_key = None

        # for i, p in enumerate(valid_moves):
        #     if next_move[1] == p[0] and next_move[0] == p[1]:
        #         action = p
        #         action_key = valid_keys[i]
        # if action_key == None:
        #     # temp_map = np.copy(self.pathfinding_map)
        #     # for i, p in enumerate(invalid_moves):
        #     #     temp_map[p[0]][p[1]] = -1
        #     #     # print(temp_map)
        #     # pathfinding_grid = Grid(matrix=temp_map)
        #     # old_distance = len(self.path_list)
        #     # self.goal_position = [valid_action_areas[self.idx][0], valid_action_areas[self.idx][1]]
        #     # start = pathfinding_grid.node(self.cur_pose[1], self.cur_pose[0])
        #     # end = pathfinding_grid.node(valid_action_areas[self.idx][1], valid_action_areas[self.idx][0])
        #     # self.path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
        #     # if len(self.path_list) <= 1:
        #     #     return None, None
        #     # self.path_list.pop(0)
        #     # self.goal_distance = (self.goal_distance - old_distance) + len(self.path_list)
        #     self.path_list = []
        #     return None, None

        # return action, action_key

    def random_move(self, valid_moves, valid_keys):
        # randomly chooses a valid move, takes in a list of valid x,y moves and the corresponding valid keys
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        choice = np.random.randint(len(valid_moves))
        a = self.action_order.index(valid_keys[choice])
        act = [0]*5
        act[a] = 1
        return valid_moves[choice], valid_keys[choice], act

    def update_next_state(self, tree_states, cur_pos):
        self.prev_state = copy.deepcopy(self.state)

        self.state = tree_states + cur_pos

    def update_next_state_path(self, tree_states, count):
        self.prev_state = copy.deepcopy(self.state)

        for i in range(1, len(tree_states)):
            self.state[i] - tree_states[i]
        self.state[self.idx] = 0

        self.state[0] = count
        self.state[-1] = self.cur_pose[1]
        self.state[-2] = self.cur_pose[0]

    # def update_next_state_path(self, tree_states, action_areas):
    #     self.prev_state = copy.deepcopy(self.state)

    #     self.state = tree_states + list(np.ndarray.flatten(np.array(valid_action_areas))) + cur_pos
    #     self.state = copy.deepcopy(tree_states)
    #     self.state.append(self.cur_pose[0])
    #     self.state.append(self.cur_pose[1])

    def update_buffer(self, reward):
        # print("PREV STATE: ", self.prev_state)
        # print("ACTION CHOICE: ", self.act)
        # print("REWARD: ", reward)
        # print("NEXT STATE ", self.state)
        # print("")
        self.policy.update_buffer(self.prev_state, self.act, reward, self.state)

    def update_buffer_shared(self, actions, reward, opposite_state):
        self.policy.update_buffer_shared(
            self.prev_state, actions, reward, self.state, opposite_state)

    def reset_agent(self):
        # TODO: Used to reset the agent after each episode
        pass


class AgentPruneSAClimited(AgentBase):
    def __init__(self, num_trees, opposite_buffer, shared_buffer, action_dim) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 200
        # Class specific action
        self.action_type = 2
        # current position
        self.cur_pose = [0, 0]
        # pathfinding
        self.action_dim = action_dim
        self.pathfinding_map = None
        self.finder = AStarFinder()
        self.goal_position = self.cur_pose
        self.old_goal_position = None
        self.start_position = self.cur_pose
        self.path_list = []
        self.goal_distance = 0

        # memory bank the size of the field.
        # initialize the learner
        self.policy = SACLimited(
            2+num_trees, action_dim, opposite_buffer, shared_buffer)
        # epsilon for exploration
        self.epsilon = 0.9
        self.state = []
        self.prev_state = []
        self.action_order = ['left', 'right', 'up', 'down', 'interact']

    def choose_move_tree(self, tree_states, valid_moves, valid_keys, cur_pos):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])

        self.state = tree_states + cur_pos

        if np.random.rand() > self.epsilon:
            action, action_key, actions = self.policy.select_action(
                tree_states, valid_moves, valid_keys, cur_pos)
            actions = actions.detach().tolist()
        else:
            action, action_key, actions = self.random_move(
                valid_moves, valid_keys)

        return action, action_key, actions

    def choose_move_tree_path(
            self, tree_states, valid_action_areas, valid_moves, valid_keys, cur_pos, invalid_moves, count):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])
        temp_map = np.copy(self.pathfinding_map)
        for i, p in enumerate(invalid_moves):
            temp_map[p[0]][p[1]] = -1
        pathfinding_grid = Grid(matrix=temp_map)

        if len(self.path_list) == 0:
            if "interact" in valid_keys:
                return valid_moves[valid_keys.index("interact")], "interact", False
            self.act = [0]*self.action_dim
            self.start_position = self.cur_pose
            #self.state = tree_states + list(np.ndarray.flatten(np.array(valid_action_areas))) + cur_pos
            self.state = [count] + tree_states + cur_pos
            self.start_position = cur_pos
            if np.random.rand() > self.epsilon:
                idx = self.policy.select_action_path(self.state)
                self.idx = idx
                self.new_goal = True
                self.goal_position = [valid_action_areas[idx][0], valid_action_areas[idx][1]]
                self.act[idx] = 1
            else:
                idx = np.random.randint(len(valid_action_areas))
                self.idx = idx
                self.new_goal = True
                self.goal_position = [valid_action_areas[idx][0], valid_action_areas[idx][1]]
                self.act[idx] = 1

        start = pathfinding_grid.node(self.cur_pose[1], self.cur_pose[0])
        end = pathfinding_grid.node(valid_action_areas[self.idx][1], valid_action_areas[self.idx][0])
        self.path_list, _ = self.finder.find_path(start, end, pathfinding_grid)
        if len(self.path_list) == 1:
            # print("HERLKJERLKEJRLKEJRLKEJR")
            self.path_list = []
            return None, None, True
        if len(self.path_list) == 0:
            # print("HERLKJERLKEJRLKEJRLKEJR")
            self.path_list = []
            return None, None, False

        self.path_list.pop(0)
        if self.new_goal:
            self.new_goal = False
            self.goal_distance = len(self.path_list)

        next_move = self.path_list.pop(0)
        action = None
        action_key = None

        for i, p in enumerate(valid_moves):
            if next_move[1] == p[0] and next_move[0] == p[1]:
                action = p
                action_key = valid_keys[i]
        if action_key == None:
            self.path_list = []
            return None, None, False

        return action, action_key, False

    def random_move(self, valid_moves, valid_keys):
        # randomly chooses a valid move, takes in a list of valid x,y moves and the corresponding valid keys
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        choice = np.random.randint(len(valid_moves))
        a = self.action_order.index(valid_keys[choice])
        act = [0]*5
        act[a] = 1
        return valid_moves[choice], valid_keys[choice], act

    def update_next_state(self, tree_states, cur_pos):
        self.prev_state = copy.deepcopy(self.state)

        self.state = tree_states + cur_pos

    # def update_next_state_path(self, tree_states):
    #     self.prev_state = copy.deepcopy(self.state)

    #     self.state = copy.deepcopy(tree_states)
    #     self.state.append(self.cur_pose[0])
    #     self.state.append(self.cur_pose[1])

    def update_next_state_path(self, tree_states, count):
        self.prev_state = copy.deepcopy(self.state)

        for i in range(1, len(tree_states)):
            self.state[i] - tree_states[i]
        self.state[self.idx] = 0

        self.state[0] = count
        self.state[-1] = self.cur_pose[1]
        self.state[-2] = self.cur_pose[0]

    # def update_next_state_path(self, tree_states):
    #     self.prev_state = copy.deepcopy(self.state)

    #     if tree_states[self.idx] == 1:
    #         self.state[self.idx] = 1
    #     else:
    #         self.state[self.idx] = 0
    #     self.state[-1] = self.cur_pose[1]
    #     self.state[-2] = self.cur_pose[0]

    def update_buffer(self, reward):
        self.policy.update_buffer(self.prev_state, self.act, reward, self.state)

    def update_buffer_shared(self, actions, reward, opposite_state):
        self.policy.update_buffer_shared(
            self.prev_state, actions, reward, self.state, opposite_state)

    def reset_agent(self):
        # TODO: Used to reset the agent after each episode
        pass

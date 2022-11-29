import numpy as np
from learning import DiscreteLearning, LearningBaseClass, SAC, SACLimited
import torch
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
    def __init__(self, field_size = [10,5]) -> None:
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
        self.memory = np.zeros([field_size[0], field_size[1],5,2])
        # intializing 0,0,0 to be our location
        self.memory[0,0,0,0] = 1
        # saving a copy of our state so we can implement state/next_state
        self.prev_state = self.memory.copy()
        # initialize the learner
        self.policy = LearningBaseClass()
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 0.9
        self.cur_cf_pose = [0,0]


    def reset_agent(self):
        # TODO: Used to reset the agent after each episode
        self.memory = np.zeros([self.field_size[0], self.field_size[1],5,2])
        # intializing 0,0,0 to be our location
        self.memory[0,0,0,0] = 1
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
                if self.memory[i,j,1] > comms[i,j,1]:
                    self.memory[i,j,:] = comms[i,j,:]
        self.comms_channel = None
        pass

    def request_communication(self, id):
        # TODO: Sets internal comms channel to the id of the robot we want a map from. This is then checked and sent over
        # in orchardsim.
        # SHOULD BE SET INTERNALLY
        self.comms_channel = id

    def choose_move(self, observed_points, observed_vals, valid_moves, valid_keys, cur_pos = None):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        if np.random.rand() > self.epsilon:
            action, action_key = self.policy.select_action(observed_points, observed_vals,valid_moves, valid_keys, cur_pos)
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
                temp_reading = [0,0,1,1,0]
            elif read == 1:
                # tree with apple
                temp_reading = [1,0,1,1,0]
            elif read == 2:
                # tree with prune
                temp_reading = [0,1,1,1,0]
            elif read == 4:
                # tree with prune and apple
                temp_reading = [1,1,1,1,0]
            elif read >= 100:
                # robot
                temp_reading = [0,0,1,0,1]
            else:
                temp_reading = [0,0,1,0,0]
            self.memory[loc[0], loc[1],:,1] = timestep
            self.memory[loc[0], loc[1],:,0] = temp_reading

    def update_epsilon(self):
        self.epsilon *= 0.999
        
    def encode_memory(self):
        #yes this is slow, ill optimize later if needed
        one_hot_memory = torch.zeros([self.field_size[0],self.field_size[1],8])
        for i,row in enumerate(self.memory):
            for j,point in enumerate(row):
                one_hot_memory[i,j,point] = 1
        return one_hot_memory
    
    def update_buffer(self, actions, reward):
        self.policy.update_buffer(self.prev_state[:,:,:,0], actions, reward, self.memory[:,:,:,0])


class AgentPickDiscrete(AgentBase):
    def __init__(self, field_size = [10,5]) -> None:
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
        self.memory = np.zeros([field_size[0], field_size[1],2])
        # intializing 0,0,0 to be our location
        self.memory[0,0,0] = 1
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
            action, action_key = self.policy.select_action_apple(appleness, valid_moves, valid_keys)
        else:
            action, action_key = self.random_move(valid_moves, valid_keys)
        
        return action, action_key
    

class AgentPruneDiscrete(AgentBase):
    def __init__(self, field_size = [10,5]) -> None:
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
        self.memory = np.zeros([field_size[0], field_size[1],2])
        # intializing 0,0,0 to be our location
        self.memory[0,0,0] = 1

        # initialize the learner
        self.policy = DiscreteLearning(field_size)
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 0.9        
        
class AgentPickSAC(AgentBase):
    def __init__(self, field_size = [10,5]) -> None:
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
        self.memory = np.zeros([field_size[0], field_size[1],5,2])
        
        self.field_size = field_size 
        # intializing 0,0,0 to be our location
        self.memory[0,0,0,0] = 1
        
        self.prev_state = self.memory.copy()
        # initialize the learner
        self.policy = SAC(field_size[0]*field_size[1]*5, 5)
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 0.9 
        self.action_order = ['left','right','up','down','interact']
        
    def choose_move(self, observed_points, observed_vals, valid_moves, valid_keys, cur_pos = None):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])
        if np.random.rand() > self.epsilon:
            action, action_key, actions = self.policy.select_action(self.memory[:,:,:,0],valid_moves, valid_keys)
            actions = actions.detach().tolist()
            cf_action, cf_action_key, cf_actions = self.random_move(valid_moves, valid_keys)
        else:
            action, action_key, actions = self.random_move(valid_moves, valid_keys)
            cf_action, cf_action_key, cf_actions = self.random_move(valid_moves, valid_keys)
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
    def __init__(self, field_size = [10,5]) -> None:
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
        self.memory = np.zeros([field_size[0], field_size[1],5,2])
        
        self.field_size = field_size 
        # intializing 0,0,0 to be our location
        self.memory[0,0,0,0] = 1
        # initialize the learner
        self.policy = SAC(field_size[0]*field_size[1]*5, 5, 'prune_agent')
        
        self.prev_state = self.memory.copy()
        # save the size of the field we are dealing with
        self.field_size = field_size
        # epsilon for exploration
        self.epsilon = 1  
        self.action_order = ['left','right','up','down','interact']

    def choose_move(self, observed_points, observed_vals, valid_moves, valid_keys, cur_pos = None):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])
        if np.random.rand() > self.epsilon:
            action, action_key, actions = self.policy.select_action(self.memory[:,:,:,0],valid_moves, valid_keys)
            actions = actions.detach().tolist()
        else:
            action, action_key, actions = self.random_move(valid_moves, valid_keys)
        
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
    def __init__(self, num_trees, name='none') -> None:
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
        # initialize the learner
        self.policy = SACLimited(2+num_trees, 5, name)
        # epsilon for exploration
        self.epsilon = 0.9 
        self.state = []
        self.prev_state = []
        self.cf_state = []
        self.action_order = ['left','right','up','down','interact']
        
    def choose_move_tree(self, tree_states, valid_moves, valid_keys, cur_pos):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])
        
        self.state = tree_states + cur_pos
        
        if np.random.rand() > self.epsilon:
            action, action_key, actions = self.policy.select_action(tree_states,valid_moves, valid_keys, cur_pos)
            actions = actions.detach().tolist()
        else:
            action, action_key, actions = self.random_move(valid_moves, valid_keys)
        
        return action, action_key, actions

    def random_move(self, valid_moves, valid_keys):
        # randomly chooses a valid move, takes in a list of valid x,y moves and the corresponding valid keys
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        choice = np.random.randint(len(valid_moves))
        a = self.action_order.index(valid_keys[choice])
        act = [0]*5
        act[a] = 1
        return valid_moves[choice], valid_keys[choice], act
    
    def update_next_state(self, tree_states, cur_pos):
        self.prev_state = self.state.copy()
        
        self.state = tree_states + cur_pos
        
    def update_cf_state(self, tree_states, cur_pos):
        self.cf_state = tree_states + cur_pos        
        
    def update_buffer(self, actions, reward, cf_reward):
        self.policy.update_buffer(self.prev_state, actions, reward, self.state, cf_reward, self.cf_state)
        
    def reset_agent(self):
        # TODO: Used to reset the agent after each episode
        pass
    
    def save_agent(self, filepath):
        self.policy.save(filepath, 'Pick')
        
class AgentPruneSAClimited(AgentBase):
    def __init__(self, num_trees, name='none') -> None:
        self.id = None
        # Class identifier
        self.robot_class = 100
        # Class specific action
        self.action_type = 2
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None
        # memory bank the size of the field.        
        # initialize the learner
        self.policy = SACLimited(2+num_trees, 5, name)
        # epsilon for exploration
        self.epsilon = 0.9 
        self.state = []
        self.prev_state = []
        self.action_order = ['left','right','up','down','interact']
        
    def choose_move_tree(self, tree_states, valid_moves, valid_keys, cur_pos):
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        # print(self.memory[:,:,2,0])
        
        self.state = tree_states + cur_pos
        
        if np.random.rand() > self.epsilon:
            action, action_key, actions = self.policy.select_action(tree_states,valid_moves, valid_keys, cur_pos)
            actions = actions.detach().tolist()
        else:
            action, action_key, actions = self.random_move(valid_moves, valid_keys)
        
        return action, action_key, actions

    def random_move(self, valid_moves, valid_keys):
        # randomly chooses a valid move, takes in a list of valid x,y moves and the corresponding valid keys
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        choice = np.random.randint(len(valid_moves))
        a = self.action_order.index(valid_keys[choice])
        act = [0]*5
        act[a] = 1
        return valid_moves[choice], valid_keys[choice], act
    
    def update_next_state(self, tree_states, cur_pos):
        self.prev_state = self.state.copy()
        
        self.state = tree_states + cur_pos
        
    def update_cf_state(self, tree_states, cur_pos):
        self.cf_state = tree_states + cur_pos 
        
    def update_buffer(self, actions, reward, cf_reward):
        self.policy.update_buffer(self.prev_state, actions, reward, self.state, cf_reward, self.cf_state)
        
    def reset_agent(self):
        # TODO: Used to reset the agent after each episode
        pass
    
    def save_agent(self, filepath):
        self.policy.save(filepath, 'Prune')
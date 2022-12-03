import numpy as np
import pygame_render
import copy
import orchard_agents


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
        self.checklist = None
        self.original_map = None

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
        # Spawn the agents at the top center of the map LENGTH NEEDS TO BE LONGER THAN NUMBER OF AGENTS
        start = (len(self.row_description) // 2) - (len(agents) // 2)
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

    def update_map(self, start: list, goal: list, key: str, agent_id: int) -> None:
        if key == "interact":
            # if we interact we update the action sequence to the next step of the goal area
            self.orchard_map[goal[0]][goal[1]] = self.action_sequence[self.orchard_map[goal[0]][goal[1]]]
        else:
            # if we move we change our previous location back to the original and update our id location
            self.orchard_map[start[0]][start[1]] = self.original_map[start[0]][start[1]]
            self.orchard_map[goal[0]][goal[1]] = agent_id

    def create_checklist(self):
        # creates a checklist containing all of the x,y location of trees to compare at the end of a timestep
        tree_checklist = []
        for i in range(np.shape(self.orchard_map)[0]):
            for j in range(len(self.row_description)):
                if self.orchard_map[i][j] in self.tree_combos:
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
        self.orchard_map = np.copy(self.original_map)
        # respawns the agents
        start = (len(self.row_description) // 2) - (len(agents) // 2)
        for i in range(len(agents)):
            self.orchard_map[0][start + i] = agents[i].robot_class
            # sets the start pose of agents and the ids
            agents[i].cur_pose = [0, start + i]


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
            for i in self.agents:
                # get valid moves for agent
                valid_moves, valid_keys = self.map.get_valid_moves(i.cur_pose, i.action_type)
                # if we have a valid move continue
                if len(valid_keys) > 0:
                    # get the surrounding area with sensors
                    points, vals = self.map.get_surroundings(i.cur_pose, 3)
                    # if internal channel is set we want to communicate
                    if i.comms_channel != None:
                        # finds the agent we want to communicate with
                        for j in self.agents:
                            if j.id == j.comms_channel:
                                # gets map from other agent
                                i.recieve_communication(j.send_communication())
                    # Agent chooses move doesnt do anything yet
                    move, key = i.choose_move(points, vals, valid_moves, valid_keys)
                    # REMOVE RANDOM MOVE ONCE CHOOSE MOVE IMPLEMENTED ONLY FOR DEMO
                    move, key = i.random_move(valid_moves, valid_keys)
                    # update our map with our action choice
                    self.map.update_map(i.cur_pose, move, key, i.id)
                    # if we moved from a spot we need to update the agents internal current position
                    if key != "interact":
                        i.cur_pose = move

                    # if we are at max timestep increment episode and reset
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

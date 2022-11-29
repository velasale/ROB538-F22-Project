import orchard_agents
import pygame_render
import orchard
import matplotlib.pyplot as plt
import table_learning as tl

# -20 is action zone, 0 is nothing, -10 is tree
large_row32 = [0, 0, -20, -10, -10, -20, 0, 0, -20, -10, -10, -20, 0,
               0, -20, -10, -10, -20, 0, 0, -20, -10, -10, -20, 0, 0,
               -20, -10, -10, -20, 0, 0]
small_row8 = [0, 0, -20, -10, -10, -20, 0, 0]

# action flow ( ex: 3 -> 2 -> -10(done) )
default_action_sequence = {1: -10, 2: -10, 3: 2, 4: 1, -10: -10}
# Action mappings, A pruner with action type 1 can do action sequences 1 or 3
default_action_map = {1: 1, 2: 2, 3: 1, 4: 2}
# probability for each tree to have an action sequence
default_prob = [.2, .1, .3, .3, .1]
# Tree representations (Mostly for GUI)
default_tree_combos = [1, 2, 3, 4, -10]

# Alejo's To check with only 1 agent and one tree
default_prob = [1, 0]
default_tree_combos = [3, -10]

# Design A:
small_row8 = [0, 0, 0, -20, -10, -20, 0, 0, 0]

# Design A:
# tstep = 500, episodes = 5000, epsilon = 0.99
# small_row8 = [0, 0, 0, -20, -10, -20, 0, 0, 0, -20, -10, -20, 0, 0, 0]


# small_row8 = [0, 0, 0, -20, -10, -20, 0, 0, -20, -10, -10, -20, 0, 0]


def large_orchard():
    # 20 agents even split between pick and prune
    # 32x15
    agent_list = []
    for i in range(10):
        a = orchard_agents.AgentPick()
        agent_list.append(a)
    for i in range(10):
        a = orchard_agents.AgentPrune()
        agent_list.append(a)
    large_orchard = orchard.OrchardMap(
        row_height=10, row_description=large_row32, top_buffer=3, bottom_buffer=2,
        action_sequence=default_action_sequence, action_map=default_action_map, tree_prob=default_prob,
        tree_combos=default_tree_combos)
    test = orchard.OrchardSim(orchard_map=large_orchard, agents=agent_list, tstep_max=100, ep_max=5)
    test.run_gui()
    # To run without GUI (Way faster)
    # test.run()


def small_orchard(approach: str):
    # 2 agents even split between pick and prune
    # 8x13

    # Grid parameters
    top_buffer = 3
    bottom_buffer = 3
    row_height = 1

    # Algorithm parameters
    time_steps = 500
    episodes = 1000

    # Create Orchard
    small_orchard = orchard.OrchardMap(
        row_height=row_height, row_description=small_row8, top_buffer=top_buffer, bottom_buffer=bottom_buffer,
        action_sequence=default_action_sequence, action_map=default_action_map, tree_prob=default_prob,
        tree_combos=default_tree_combos)

    # Create Agents
    rows = row_height + top_buffer + bottom_buffer
    cols = len(small_row8)
    agent_list = []

    for i in range(1):
        a = orchard_agents.AgentPick(rows, cols)
        agent_list.append(a)
    for i in range(1):
        a = orchard_agents.AgentPrune(rows, cols)
        agent_list.append(a)

    test = orchard.OrchardSim(orchard_map=small_orchard, agents=agent_list, tstep_max=time_steps, ep_max=episodes)
    # test.run_gui()
    # To run without GUI (Way faster)
    test.run(approach)

    return test


if __name__ == "__main__":

    approach = "diff"
    test = small_orchard(approach)
    # test_random = small_orchard("random")

    for i in range(len(test.agents)):
        tl.plot_reward(test.agents[i].reward_evolution, i)
        # tl.plot_reward_and_baseline(test.agents[i].reward_evolution, test_random.agents[i].reward_evolution, i)
        tl.plot_values(test.agents[i].q_sa_table, i)
    plt.title(approach)
    plt.show()
    a=1
    # test = large_orchard()

import orchard_agents
import pygame_render
import orchard

# -20 is action zone, 0 is nothing, -10 is tree
large_row32 = [0, 0, -20, -10, -10, -20, 0, 0, -20, -10, -10, -20, 0,
               0, -20, -10, -10, -20, 0, 0, -20, -10, -10, -20, 0, 0,
               -20, -10, -10, -20, 0, 0]
small_row8 = [0, 0, -20, -10, -10, -20, 0, 0]


# action flow ( ex: 3 -> 2 -> -10(done) )
default_action_sequence = {1: -10, 2: -10, 3: 2, 4: 1}
# Action mappings, A pruner with action type 1 can do action sequences 1 or 3
default_action_map = {1: 1, 2: 2, 3: 1, 4: 2}
# probability for each tree to have an action sequence
default_prob = [.4, .1, .1, .3, .1]
# Tree representations (Mostly for GUI)
default_tree_combos = [1, 2, 3, 4, -10]


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


def small_orchard():
    # 2 agents even split between pick and prune
    # 8x13
    agent_list = []
    for i in range(1):
        a = orchard_agents.AgentPick()
        agent_list.append(a)
#    for i in range(1):
        #a = orchard_agents.AgentPrune()
        # agent_list.append(a)
    small_orchard = orchard.OrchardMap(
        row_height=8, row_description=small_row8, top_buffer=3, bottom_buffer=2,
        action_sequence=default_action_sequence, action_map=default_action_map, tree_prob=default_prob,
        tree_combos=default_tree_combos, num_classes=1)
    test = orchard.OrchardSim(orchard_map=small_orchard, agents=agent_list, tstep_max=50, ep_max=200)
    test.run_gui()
    # To run without GUI (Way faster)
    # test.run()


if __name__ == "__main__":
    test = small_orchard()
    #test = large_orchard()

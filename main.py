import orchard_agents
# import pygame_render
import orchard
import matplotlib.pyplot as plt
import numpy as np

# -20 is action zone, 0 is nothing, -10 is tree
large_row32 = [0, 0, -20, -10, -10, -20, 0, 0, -20, -10, -10, -20, 0,
               0, -20, -10, -10, -20, 0, 0, -20, -10, -10, -20, 0, 0,
               -20, -10, -10, -20, 0, 0]
small_row8 = [-20, -10, -10, -20, -20, -10, -10, -20]


# action flow ( ex: 3 -> 2 -> -10(done) )
default_action_sequence = {1: -10, 2: -10, 3: 2, 4: 1}
# Action mappings, A pruner with action type 1 can do action sequences 1 or 3
default_action_map = {1: 1, 2: 2, 3: 1, 4: 2}
# probability for each tree to have an action sequence
default_prob = [.2, .1, .3, .3, .1]
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
    # test.run_gui()
    # To run without GUI (Way faster)
    test.run()


def small_orchard():
    # 2 agents even split between pick and prune
    # 8x13
    test_name = 'Local_'
    agent_list = []
    for i in range(1):
        a = orchard_agents.AgentPickSAClimited(40)
        b = orchard_agents.AgentPruneSAClimited(40)
        agent_list.append(a)
        agent_list.append(b)

    small_orchard = orchard.OrchardMap(
        row_height=10, row_description=small_row8, top_buffer=1, bottom_buffer=1,
        action_sequence=default_action_sequence, action_map=default_action_map, tree_prob=default_prob,
        tree_combos=default_tree_combos)
    num_eps = 1000
    test = orchard.OrchardSim(orchard_map=small_orchard, agents=agent_list, tstep_max=100, ep_max=num_eps)
    # test.run_gui()
    # To run without GUI (Way faster)
    
    test.run()
    for agents in test.agents:
        agents.save_agent(test_name)
    test.map.save_data(test_name+'data')
    g_rewards = np.array([i[-1] for i in test.map.episode_global_rewards])
    plt.plot(range(num_eps),g_rewards)
    plt.title('Episode Global Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Pruned*Picked)')
    plt.show()
    plt.clf()
    
    num_rewards = np.array(test.map.rewards)
    num_trees = np.array(test.map.num_trees)
    plt.plot(range(num_eps),num_rewards[:,0]/num_trees[:-1,0]*100)
    plt.plot(range(num_eps),num_rewards[:,1]/num_trees[:-1,1]*100)
    plt.legend(['Picked Apples','Pruned Trees'])
    plt.title('Picked and Pruned Apples per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Percent of Relevant Trees Interacted with')
    plt.show()

    


def small_orchard_single():
    # 2 agents even split between pick and prune
    # 8x13
    test_name = 'Local'
    agent_list = []
    for i in range(1):
        a = orchard_agents.AgentPickSAClimited(40)
        agent_list.append(a)

    small_orchard = orchard.OrchardMap(
        row_height=10, row_description=small_row8, top_buffer=1, bottom_buffer=1,
        action_sequence=default_action_sequence, action_map=default_action_map, tree_prob=default_prob,
        tree_combos=default_tree_combos)
    num_eps = 10
    test = orchard.OrchardSim(orchard_map=small_orchard, agents=agent_list, tstep_max=70, ep_max=num_eps)
    # test.run_gui()
    # To run without GUI (Way faster)
    test.run()
    for agents in test.agents:
        agents.save_agent(test_name)
    test.map.save_data(test_name+'data')
    num_rewards = [sum(rew) for rew in test.map.rewards]
    print(num_rewards)
    plt.plot(range(num_eps),num_rewards)
    plt.show()
    plt.clf()
    plt.plot(range(len(test.map.rewards[0])),test.map.rewards[0])
    plt.plot(range(len(test.map.rewards[-1])),test.map.rewards[-1])
    plt.legend(['first','last'])
    
if __name__ == "__main__":
    test = small_orchard()
    # test = large_orchard()

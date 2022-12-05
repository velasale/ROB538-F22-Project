import orchard_agents
# import pygame_render
import orchard
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from replaybuffer import ReplayBuffer

# -20 is action zone, 0 is nothing, -10 is tree
large_row32 = [0, 0, -20, -10, -10, -20, 0, 0, -20, -10, -10, -20, 0,
               0, -20, -10, -10, -20, 0, 0, -20, -10, -10, -20, 0, 0,
               -20, -10, -10, -20, 0, 0]
small_row8 = [0, -20, -10, -10, -20, 0, -20, -10, -10, -20, 0]


# action flow ( ex: 3 -> 2 -> -10(done) )
default_action_sequence = {1: -10, 2: -10, 3: 2, 4: 1}
# Action mappings, A pruner with action type 1 can do action sequences 1 or 3
default_action_map = {1: 1, 2: 2, 3: 1, 4: 2}
# probability for each tree to have an action sequence
default_prob = [.2, .1, .3, .3, .1]
# Tree representations (Mostly for GUI)
default_tree_combos = [1, 2, 3, 4, -10]


def create_pathfinding_map(orch, row_d):
    tree_checklist = np.copy(orch)
    for i in range(np.shape(orch)[0]):
        for j in range(len(row_d)):
            if orch[i][j] == -20 or orch[i][j] == 0 or orch[i][j] == 100 or orch[i][j] == 200:
                tree_checklist[i][j] = 1
            if orch[i][j] in default_tree_combos:
                tree_checklist[i][j] = -1
    return tree_checklist


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
    test = orchard.OrchardSim(
        orchard_map=large_orchard, agents=agent_list, tstep_max=150, ep_max=5)
    # test.run_gui()
    # To run without GUI (Way faster)
    test.run()


def small_orchard():
    # 2 agents even split between pick and prune
    # 8x13
    agent_list = []
    shared_a = ReplayBuffer()
    shared_b = ReplayBuffer()
    for i in range(1):
        a = orchard_agents.AgentPickSAClimited(
            41, opposite_buffer=shared_a, shared_buffer=shared_b, action_dim=40)
        b = orchard_agents.AgentPruneSAClimited(
            41, opposite_buffer=shared_b, shared_buffer=shared_a, action_dim=40)
        agent_list.append(a)
        agent_list.append(b)

    # large_orchard = orchard.OrchardMap(
    #     row_height=5, row_description=large_row32, top_buffer=3, bottom_buffer=2,
    #     action_sequence=default_action_sequence, action_map=default_action_map, tree_prob=default_prob,
    #     tree_combos=default_tree_combos)
    small_orchard = orchard.OrchardMap(
        row_height=10, row_description=small_row8, top_buffer=2, bottom_buffer=2,
        action_sequence=default_action_sequence, action_map=default_action_map, tree_prob=default_prob,
        tree_combos=default_tree_combos, seed=1253)

    pm = create_pathfinding_map(small_orchard.orchard_map, small_row8)
    small_orchard.pathfinding_map = pm

    small_orchard.create_map(agent_list)
    for i in range(len(agent_list)):
        agent_list[i].pathfinding_map = create_pathfinding_map(small_orchard.orchard_map, small_row8)
        print(agent_list[i].cur_pose)

    num_eps = 3000
    test = orchard.OrchardSim(
        orchard_map=small_orchard, agents=agent_list, tstep_max=300, ep_max=num_eps)
    #  est.run_gui()
    # To run without GUI (Way faster)
    test.run()
    fig = plt.figure()
    average_data = []
    average_data2 = []
    average_data3 = []
    window = 50
    color = 'blue'
    color2 = 'orange'
    color3 = 'red'

    num_rewards = np.array(test.map.rewards)
    rewards_evolution = num_rewards[:, 0]

    # Step 1: Obtain the moving average window
    for ind in range(len(rewards_evolution) - window + 1):
        average_data.append(np.mean(rewards_evolution[ind:ind + window]))
    label = "Percent of Orchard Complete (MA)"

    # rewards_evolution2 = num_rewards[:, 1]
    # for ind in range(len(rewards_evolution2) - window + 1):
    #     average_data2.append(np.mean(rewards_evolution2[ind:ind + window]))
    # label = "Percent of Apples Picked (MA)"

    # rewards_evolution3 = num_rewards[:, 2]
    # for ind in range(len(rewards_evolution3) - window + 1):
    #     average_data3.append(np.mean(rewards_evolution3[ind:ind + window]))
    # label = "Percent of Trees Pruned (MA)"
    save_dict = {"total": num_rewards[:, 0], "pick": num_rewards[:, 1], "prune": num_rewards[:, 2]}
    with open("globallocal_pathfinding_1253.pkl", "wb+") as file:
        pkl.dump(save_dict, file)
    # Step 2: Plot results
    plt.plot(average_data, color=color, label=label)
    #plt.plot(average_data2, color=color2, label=label)
    #plt.plot(average_data3, color=color3, label=label)
    # alpha sets the transparency of the original data
    plt.plot(rewards_evolution, color=color, alpha=0.2, label='Total Percent')
    #plt.plot(rewards_evolution2, color=color2, alpha=0.2, label='Pick Percent')
    #plt.plot(rewards_evolution3, color=color3, alpha=0.2, label='Prune Percent')
    plt.xlabel("Episodes")
    plt.ylabel("Percent Complete")
    plt.legend()
    plt.title("Rollback Buffer Reward: Rollback = 1, Rollback Decay: .5")

    plt.show()
    plt.clf()
    plt.plot(range(len(test.map.rewards[0])), test.map.rewards[0])
    plt.plot(range(len(test.map.rewards[-1])), test.map.rewards[-1])
    plt.legend(['first', 'last'])


def small_orchard_single():
    # 2 agents even split between pick and prune
    # 8x13
    agent_list = []
    for i in range(1):
        a = orchard_agents.AgentPickSAClimited(40)
        agent_list.append(a)

    small_orchard = orchard.OrchardMap(
        row_height=10, row_description=small_row8, top_buffer=1, bottom_buffer=1,
        action_sequence=default_action_sequence, action_map=default_action_map, tree_prob=default_prob,
        tree_combos=default_tree_combos)

    for i in len(agent_list):
        agent_list[i].pathfinding_map = create_pathfinding_map

    num_eps = 100
    test = orchard.OrchardSim(
        orchard_map=small_orchard, agents=agent_list, tstep_max=200, ep_max=num_eps)
    test.run_gui()
    # To run without GUI (Way faster)
    # test.run()
    num_rewards = [sum(rew) for rew in test.map.rewards]
    print(num_rewards)
    plt.plot(range(num_eps), num_rewards)
    plt.show()
    plt.clf()
    plt.plot(range(len(test.map.rewards[0])), test.map.rewards[0])
    plt.plot(range(len(test.map.rewards[-1])), test.map.rewards[-1])
    plt.legend(['first', 'last'])


if __name__ == "__main__":
    test = small_orchard()
    # test = large_orchard()

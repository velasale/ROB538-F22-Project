import matplotlib.pyplot as plt
import numpy as np

# TODO improve Global Reward formula


def random_learning(agents, map):
    """
    Random Learning to use as a Baseline
    Simply keep epsilon = 1
    :param agents:
    :param map:
    :param steps:
    :return:
    """

    # main control loop for all agents
    for i in agents:

        i.epsilon = 1

        # get valid moves for agent
        valid_moves, valid_keys = map.get_valid_moves(i.cur_pose, i.action_type, i.id)
        # if we have a valid move continue
        if len(valid_keys) > 0:
            # get the surrounding area with sensors
            points, vals = map.get_surroundings(i.cur_pose, 3)
            # print("Valid moves and valid keys are: ", valid_moves, valid_keys)

            # --- Step 1: Take next action ---
            # a - Observe next state
            move, key = i.choose_move(points, vals, valid_moves, valid_keys)

            # b. Observe reward
            reward = map.reward_map[move[0]][move[1]]

            # --- Step 2: Choose A_prime from S_prime
            valid_moves_prime, valid_keys_prime = map.get_valid_moves(move, i.action_type, i.id)
            move_2, key_2 = i.choose_move(points, vals, valid_moves_prime, valid_keys_prime)

            # --- Step 3: Update Q_sa_values of the agent
            i.update_value(move, move_2, reward)
            i.accumulated_reward = i.accumulated_reward + reward

            # update our map with our action choice
            map.update_map(i.cur_pose, move, key, i.id)
            # if we moved from a spot we need to update the agents internal current position
            if key != "interact":
                i.cur_pose = move

            # Update epsilon
            # i.update_epsilon()

            if reward == 10:
                # Update reward map
                map.reward_map[move[0]][move[1]] = -1

    return agents, map


def local_rewards(agents, map):
    """
    Assumes that agents' actions are independent.
    Each agent has its own Q-learning Temporal Difference
    :param agents:
    :param map:
    :param steps:
    :return:
    """

    # main control loop for all agents
    for i in agents:
        # get valid moves for agent
        valid_moves, valid_keys = map.get_valid_moves(i.cur_pose, i.action_type, i.id)
        # if we have a valid move continue
        if len(valid_keys) > 0:
            # get the surrounding area with sensors
            points, vals = map.get_surroundings(i.cur_pose, 3)
            # print("Valid moves and valid keys are: ", valid_moves, valid_keys)

            # --- Step 1: Take next action ---
            # a - Observe next state
            move, key = i.choose_move(points, vals, valid_moves, valid_keys)

            # b. Observe reward
            reward = map.reward_map[move[0]][move[1]]

            # --- Step 2: Choose A_prime from S_prime
            valid_moves_prime, valid_keys_prime = map.get_valid_moves(move, i.action_type, i.id)
            move_2, key_2 = i.choose_move(points, vals, valid_moves_prime, valid_keys_prime)

            # --- Step 3: Update Q_sa_values of the agent
            i.update_value(move, move_2, reward)
            i.accumulated_reward = i.accumulated_reward + reward

            # update our map with our action choice
            map.update_map(i.cur_pose, move, key, i.id)
            # if we moved from a spot we need to update the agents internal current position
            if key != "interact":
                i.cur_pose = move

            # Update epsilon
            i.update_epsilon()

            if reward == 10:
                # Update reward map
                map.reward_map[move[0]][move[1]] = -1

    return agents, map


def global_rewards(agents, map):
    """
    Assumes that agents' actions are independent, hence each agent has its own Q-learning Temporal Difference
    The system/global reward is kept based on the sum of the rewards from all agents.



    Temporal Difference - Global Rewards
    :param agents:
    :param map:
    :param steps:
    :return:
    """

    # main control loop for all agents
    for i in agents:
        # get valid moves for agent
        valid_moves, valid_keys = map.get_valid_moves(i.cur_pose, i.action_type, i.id)
        # if we have a valid move continue
        if len(valid_keys) > 0:
            # get the surrounding area with sensors
            points, vals = map.get_surroundings(i.cur_pose, 3)
            # print("Valid moves and valid keys are: ", valid_moves, valid_keys)

            # --- Step 1: Take next action ---
            # a - Observe next state
            i.move, i.key = i.choose_move(points, vals, valid_moves, valid_keys)

            # b. Observe reward
            i.reward = map.reward_map[i.move[0]][i.move[1]]

            # --- Step 2: Choose A_prime from S_prime
            valid_moves_prime, valid_keys_prime = map.get_valid_moves(i.move, i.action_type, i.id)
            i.move_2, key_2 = i.choose_move(points, vals, valid_moves_prime, valid_keys_prime)

    global_reward = 0
    for i in agents:
        # global_reward += i.accumulated_reward
        global_reward += i.reward

    for i in agents:
        # --- Step 3: Update Q_sa_values of the agent
        i.update_value(i.move, i.move_2, global_reward)
        i.accumulated_reward = i.accumulated_reward + i.reward

        # update our map with our action choice
        map.update_map(i.cur_pose, i.move, i.key, i.id)
        # if we moved from a spot we need to update the agents internal current position
        if i.key != "interact":
            i.cur_pose = i.move

        # Update epsilon
        i.update_epsilon()

        if i.reward == 10:
            # Update reward map
            map.reward_map[i.move[0]][i.move[1]] = -1

    return agents, map


def diff_rewards(agents, map):
    """
    Temporal Difference - Global Rewards
    :param agents:
    :param map:
    :param steps:
    :return:
    """

    # main control loop for all agents
    for i in agents:
        # get valid moves for agent
        valid_moves, valid_keys = map.get_valid_moves(i.cur_pose, i.action_type, i.id)
        # if we have a valid move continue
        if len(valid_keys) > 0:
            # get the surrounding area with sensors
            points, vals = map.get_surroundings(i.cur_pose, 3)
            # print("Valid moves and valid keys are: ", valid_moves, valid_keys)

            # --- Step 1: Take next action ---
            # a - Observe next state
            i.move, i.key = i.choose_move(points, vals, valid_moves, valid_keys)

            # b. Observe reward
            i.reward = map.reward_map[i.move[0]][i.move[1]]

            # --- Step 2: Choose A_prime from S_prime
            valid_moves_prime, valid_keys_prime = map.get_valid_moves(i.move, i.action_type, i.id)
            i.move_2, key_2 = i.choose_move(points, vals, valid_moves_prime, valid_keys_prime)

    global_reward = 0
    for i in agents:
        global_reward += i.reward

    for i in agents:
        # Obtain the Counterfactual reward
        others_reward = 0
        for j in agents:
            if j != i:
                others_reward = others_reward + j.reward

        diff_reward = global_reward - others_reward

        # --- Step 3: Update Q_sa_values of the agent
        i.update_value(i.move, i.move_2, diff_reward)
        i.accumulated_reward = i.accumulated_reward + diff_reward

        # update our map with our action choice
        map.update_map(i.cur_pose, i.move, i.key, i.id)
        # if we moved from a spot we need to update the agents internal current position
        if i.key != "interact":
            i.cur_pose = i.move

        # Update epsilon
        i.update_epsilon()

        if i.reward == 10:
            # Update reward map
            map.reward_map[i.move[0]][i.move[1]] = -1

    return agents, map


def plot_reward(rewards_evolution: list, i):
    """
    Plots RL rewards evolution with a shaded background
    :param rewards_evolution:
    :param i:
    :return:
    """
    fig = plt.figure()
    average_data = []
    window = 50
    color = 'blue'

    # Step 1: Obtain the moving average window
    for ind in range(len(rewards_evolution) - window + 1):
        average_data.append(np.mean(rewards_evolution[ind:ind + window]))
    label = "Moving Average with " + str(window) + " window"

    # Step 2: Plot results
    plt.plot(average_data, color=color, label=label)
    # alpha sets the transparency of the original data
    plt.plot(rewards_evolution, color=color, alpha=0.25, label='Original Rewards')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.title(i)


def plot_values(values: list, i):
    """Plots values for each state
    """
    # --- Q_value table
    fig = plt.figure()
    plt.imshow(values)
    plt.colorbar()
    plt.title(i)
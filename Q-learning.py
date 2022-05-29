import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 环境
HEIGHT = 4
WIDTH = 12

EPSILON = 0.1

ALPHA = 0.5

GAMMA = 0.7

# 动作空间
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

START = [3, 0]
GOAL = [3, 11]

q_sarsa = np.zeros((HEIGHT, WIDTH, 4))
q_q_learning = np.copy(q_sarsa)

def move(state, action):
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WIDTH - 1)]
    else:
        next_state = [min(i + 1, HEIGHT - 1), j]

    reward = -1
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
        action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward

def epsilon_greedy(eps, q_value, state):
    if np.random.binomial(1, eps) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    return action

def Sarsa(q_value):
    state = START
    action = epsilon_greedy(EPSILON, q_value, state)
    rewards = 0.0
    while state != GOAL:
        next_state, reward = move(state, action)
        next_action = epsilon_greedy(EPSILON, q_value, next_state)
        rewards += reward
        # target = q_value[next_state[0], next_state[1], next_action]

        q_value[state[0], state[1], action] += \
            ALPHA * ( reward + q_value[next_state[0], next_state[1], next_action] -
                               q_value[state[0], state[1], action])
        state = next_state
        action = next_action
    return rewards

def Q_Learning(q_value):
    state = START
    rewards = 0.0
    while state != GOAL:
        action = epsilon_greedy(EPSILON, q_value, state)
        next_state, reward = move(state, action)
        rewards += reward
        q_value[state[0], state[1], action] += ALPHA * (
                reward + np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
    return rewards

# 最优策略
def optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, HEIGHT):
        optimal_policy.append([])
        for j in range(0, WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append(9)
                continue
            bestAction = np.argmax(q_value[i, j, :])
            optimal_policy[-1].append(bestAction)

    return np.array(optimal_policy)

def print_op(optimal_policy):
    for row in optimal_policy:
        print(row)

def train():
    episodes = 1000
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    for r in tqdm(range(runs)):
        for i in range(0, episodes):
            rewards_sarsa[i] += Sarsa(q_sarsa)
            rewards_q_learning[i] += Q_Learning(q_q_learning)

    rewards_sarsa /= runs
    rewards_q_learning /= runs

    # draw reward curves
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()
    plt.show()
    # plt.savefig('../images/figure_6_4.png')
    plt.close()

    return optimal_policy(q_sarsa), optimal_policy(q_q_learning)


def test(OP):
    state = START
    rewards = 0
    while state != GOAL:
        action = OP[state[0], state[1]]
        next_state, reward = move(state, action)
        rewards += reward
        state = next_state
    return rewards

# Due to limited capacity of calculation of my machine, I can't complete this experiment
# with 100,000 episodes and 50,000 runs to get the fully averaged performance
# However even I only play for 1,000 episodes and 10 runs, the curves looks still good.
def figure_6_6():
    step_sizes = np.arange(0.1, 1.1, 0.1)
    episodes = 1000
    runs = 10

    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_QLEARNING = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_QLEARNING = 5
    methods = range(0, 6)

    performace = np.zeros((6, len(step_sizes)))
    for run in range(runs):
        for ind, step_size in tqdm(list(zip(range(0, len(step_sizes)), step_sizes))):
            q_sarsa = np.zeros((HEIGHT, WIDTH, 4))
            q_expected_sarsa = np.copy(q_sarsa)
            q_q_learning = np.copy(q_sarsa)
            for ep in range(episodes):
                sarsa_reward = Sarsa(q_sarsa)
                # expected_sarsa_reward = Sarsa(q_expected_sarsa, expected=True, step_size=step_size)
                q_learning_reward = Q_Learning(q_q_learning)
                performace[ASY_SARSA, ind] += sarsa_reward
                # performace[ASY_EXPECTED_SARSA, ind] += expected_sarsa_reward
                performace[ASY_QLEARNING, ind] += q_learning_reward

                if ep < 100:
                    performace[INT_SARSA, ind] += sarsa_reward
                    # performace[INT_EXPECTED_SARSA, ind] += expected_sarsa_reward
                    performace[INT_QLEARNING, ind] += q_learning_reward

    performace[:3, :] /= episodes * runs
    performace[3:, :] /= 100 * runs
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']

    for method, label in zip(methods, labels):
        plt.plot(step_sizes, performace[method, :], label=label)
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()
    plt.show()
    plt.close()

if __name__ == '__main__':
    opS, opQ = train()

    # display optimal policy
    print('Sarsa Optimal Policy:')
    print_op(opS)
    print('Sarsa Control:')
    print(test(opS))
    print('Q-Learning Optimal Policy:')
    print_op(opQ)
    print('Q-Learning Control:')
    print(test(opQ))


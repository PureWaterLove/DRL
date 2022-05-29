
import numpy as np

# 环境
HEIGHT = 7 # 高
WIDTH = 10 # 宽
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] # wind strength for each column 每一列风力大小

# 动作空间
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

START = [3, 0] #初始状态，最左上角是【0,0】最右下角是【6,9】
GOAL = [3, 7]

EPSILON = 0.1
ALPHA = 0.5
REWARD = -1.0

# 移动
def move(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j] #纵向上：向上移动一格，也要受到风的影响，最小是最上面，即0。横向上：不变
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], HEIGHT - 1), 0), j] #纵向上：向下移动一格，也要受到风的影响，最小是最上面，即0；最大是最下面，即高度-1。横向上：不变
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)] #纵向上：受到风的影响，最小是最上面，即0。横向上：向左运动，最小是最左面，即0
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WIDTH - 1)] #纵向上：受到风的影响，最小是最上面，即0。横向上：向右运动，最大是最右面，即宽度-1
    else:
        assert False

def epsilon_greedy(eps, q_value, state):
    if np.random.binomial(1, eps) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    return action

# Sarsa算法
def Sarsa(q_value):
    state = START # 初始化状态
    action = epsilon_greedy(EPSILON, q_value, state) # 在状态中，基于Q并使用贪婪策略选择动作

    while state != GOAL: #循环执行直到终止
        next_state = move(state, action) # 更新状态
        next_action = epsilon_greedy(EPSILON, q_value, state) # 在状态中，基于Q并使用贪婪策略更新动作

        # 使用公式更新Q
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        state = next_state #将新状态与新动作存为当前状态与当前动作
        action = next_action

def optimal_policy():
    q_value = np.zeros((HEIGHT, WIDTH, 4))
    episode_limit = 5000
    ep = 0
    while ep < episode_limit:
        Sarsa(q_value)
        ep += 1

    optimal_policy = []
    for i in range(0, HEIGHT):
        optimal_policy.append([])
        for j in range(0, WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)

if __name__ == '__main__':
    optimal_policy()



import numpy as np
import matplotlib.pyplot as plt
from gym import make

def transform_state(state):
    x, y = state
    return np.array(((x + 1.2) / 1.8, (y + 0.07) / 0.14))
     

def evaluate(policy, episodes, seed=42):
    env = make("MountainCar-v0")
    env.seed(seed)
    env.action_space.seed(seed)
    returns = []
    for i in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        while not done:
            state = transform_state(state)
            state, reward, done, _ = env.step(policy.action(state))
            total_reward += reward
        returns.append(total_reward)
    env.close()
    return returns

def phase_space(size_x, size_y, policy):
    """
    Action that policy takes in some points
    of phase spac.
    
    Axis Х -- state[0] (position)
    Axis Y -- state[1] (velocity)
    
    Red point color -- action == 0(acceleration backward)
    Green poin color -- action == 1(no acceleration)
    Blue point color -- action == 2(acceleration forward)
    """
    X = np.linspace(-1.2, 0.6, size_x)
    Y = np.linspace(-0.07, 0.07, size_y)
    col = ['ro', 'go', 'bo']
    D = {}
    for x in X:
        for y in Y:
            point = transform_state((x, y))
            D[(x, y)] = policy.action(point)
    for k, v in D.items():
        plt.plot([k[0]], [k[1]], col[v])
    plt.title("Action taken by policy")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.show()

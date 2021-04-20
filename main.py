from gym import make
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
from time import perf_counter as pc
import random

from algorithm import REINFORCE
from utilities import transform_state, evaluate, phase_space


if __name__ == "__main__":

    START = pc()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    env = make("MountainCar-v0")
    env.seed(0)
    env.action_space.seed(0)
    
    policy = REINFORCE(input_dim=2, hidden_dim=16, action_dim=3, max_len=200)

    number_of_updates = 1_200
    number_of_trajectories =16
    Means = []
    Std = []
    
    for i in range(number_of_updates):
        trajectory_batch = []
        
        # Get batch
        for t in range(number_of_trajectories):
            done = False
            state = env.reset()
            trajectory_batch.append([])
            #time_step = 0
            
            while not done:
                state = transform_state(state)
                action  = policy.action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Change the reward, otherwise its constant -1
                # Uncomment penalty and `time_step` lines to make use of time agent as alive
                reward = 2 * abs(next_state[1]) / 0.07  # - (time_step / 500)
                
                trajectory_batch[-1].append((state, action, next_state, reward, done))
                state = next_state
                #time_step += 1
                
        # Update
        policy.update(trajectory_batch)
        
        # Evaluate
        if (i + 1) % 20 == 0:
            rewards = evaluate(policy, 30)
            Means.append(np.mean(rewards))
            Std.append(np.std(rewards))
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Time: {int(pc() - START) // 60} min {int(pc() - START) % 60} sec")
            
    # Plot mean rewards
    Means = np.array(Means)
    Std = np.array(Std)
    Time = np.arange(0, number_of_updates, 20)

    plt.fill_between(Time, Means - Std, Means + Std, alpha=0.3)
    plt.plot(Time, Means, linewidth = 3)
    plt.grid()
    plt.title("Mean reward")
    plt.ylabel("Reward")
    plt.xlabel("Number of batches")
    plt.show()

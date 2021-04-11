from gym import make
import numpy as np
import copy
import torch
from torch import nn
import random
from torch.optim import Adam

learning_rate = 5e-3

class REINFORCE:
    def __init__(self, input_dim, hidden_dim, action_dim):
        self.model = nn.Sequential( 
                     nn.Linear(input_dim, hidden_dim),
                     nn.ReLU(), 
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.ReLU(),
                     nn.Linear(hidden_dim, action_dim))
                     
        self.optimizer = Adam(self.model.parameters(), learning_rate)
                     
        

    def update(self, trajectories):
        
        total_sum = torch.Tensor(len(trajectories)) 
        for j,traj in enumerate(trajectories):
            
            # Take cumulative sum
            rewards = torch.tensor([moment[3] for moment in reversed(traj)])
            cum_reward = torch.cumsum(rewards, dim=0)
            cum_reward = torch.flip(cum_reward, [0])
            
            # Take logarithm of policy probabilites
            output = self.model(torch.tensor([moment[0] for moment in traj]).float())
            prob_output = nn.Softmax()(output)
            prob_log = torch.Tensor(len(traj))
            for i in range(len(traj)):
                prob_log[i] = prob_output[(i, traj[i][1])]
            prob_log = torch.log(prob_log)
            
            # Sum on all trajectrory
            total_sum[j] = torch.sum(prob_log * cum_reward)
            
        # Backward pass
        loss = -torch.mean(total_sum)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
    def action(self, eps, env, state):
        if random.random() < eps:
            return env.action_space.sample()
        state = torch.tensor(state).unsqueeze(0).float()
        ac = self.model(state)
        return torch.argmax(ac).numpy()

        
def evaluate(policy, episodes, eps):
    env = make("MountainCar-v0")
    returns = []
    for i in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        while not done:
            state, reward, done, _ = env.step(policy.action(eps, env, state))
            total_reward += reward
        returns.append(total_reward)
    env.close()
    return returns


if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    env = make("MountainCar-v0")
    env.seed(0)
    env.action_space.seed(0)
    
    policy = REINFORCE(input_dim=2, hidden_dim=16, action_dim=3)
     
    eps = 0.3
    number_of_updates = 4_000_000
    number_of_trajectories = 32
    
    for i in range(number_of_updates):
        trajectory_batch = []
        
        for t in range(number_of_trajectories):
            done = False
            state = env.reset()
            trajectory_batch.append([])
            
            while not done:
                action = policy.action(eps, env, state)
                next_state, reward, done, _ = env.step(action)
                
                reward += abs(next_state[1])/ 0.07 # Reward shaping
                trajectory_batch[-1].append((state, action, next_state, reward, done))
                state = next_state
                
        policy.update(trajectory_batch)
        
        if (i + 1) % 100 == 0:
       
            eps = 0.25 * (1 - i / number_of_updates) + 0.03  # Reduce exploration with time
            rewards = evaluate(policy, 30, eps)
              
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")

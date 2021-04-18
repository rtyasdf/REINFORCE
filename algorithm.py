from torch import nn
from torch.optim import Adam
import torch
import numpy as np

learning_rate = 1e-2
GAMMA = 0.95

class REINFORCE:
    def __init__(self, input_dim, hidden_dim, action_dim, max_len):
        self.model = nn.Sequential( 
                     nn.Linear(input_dim, hidden_dim),
                     nn.ReLU(), 
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.ReLU(),
                     nn.Linear(hidden_dim, action_dim),
                     nn.Softmax(dim=-1))
        
        self.action_dim = action_dim
        self.optimizer = Adam(self.model.parameters(), learning_rate)
        self.gammas = torch.tensor([np.power(GAMMA, i) for i in range(max_len)])
        
        

    def update(self, trajectories):
        
        total_sum = torch.Tensor(len(trajectories))
        for j,traj in enumerate(trajectories):
                
            # Take cumulative rewards with discount factor
            rewards = torch.tensor([moment[3] for moment in reversed(traj)])
            gammas_reversed = torch.flip(self.gammas[:len(traj)], [0])
            
            cum_reward = torch.cumsum(rewards * gammas_reversed, dim=0)
            cum_reward = torch.flip(cum_reward, [0])
            cum_reward = cum_reward / self.gammas[:len(traj)]
            
            
            # Get probabilities
            output = self.model(torch.tensor([moment[0] for moment in traj]).float())
            
            # Take logarithm of probabilities
            prob_ac = torch.Tensor(len(traj))
            for i,m in enumerate(traj):
                prob_ac[i] = output[i, m[1]]
            prob_log = torch.log(prob_ac)
            
            # Get loss
            total_sum[j] = torch.sum(prob_log * cum_reward)
            
        # Take a gradient step
        loss = -torch.mean(total_sum)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        
    def action(self, state):
        state = torch.tensor(state).float()
        distribution = self.model(state)
        ac = np.random.choice(range(self.action_dim), p=distribution.detach().numpy())
        return ac

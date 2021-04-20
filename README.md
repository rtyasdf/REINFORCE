# REINFORCE
Implementation of reinforce algorithm for mountain-car environment.

## Policy
Policy is based on neural network with 2 neurons on input, 2 hidden linear layers with 16 neurons and relu activation, 3 neurons on output and softmax operation to take probabilities of actions. Action on each step taken proportionallly to corresponding probabilities. Neural network updated on loss from 16 trajectories sampled sequentially.

## Reward
Because in mountain car environment reward for each step is constant -1 and sum of rewards is multiplier for loss, reward on each step is changed to normalized velocity. Without such change or reward shaping as alternative it'll be impossible to learn something. On each step we can substract a penalty for time agent is alive. Comparison of total gained reward can be found in **mean_reward.png**.

## Parameters
As was mentioned above each batch consist of 16 rewards, discount factor equals 0.95, total number of batches 1200, although it's possible to go beyond and policy begin to gain reward after 320-340 batches. For policy neural network Adam optimizer was used with learning rate 0.01.

## Loss
Loss for reinforce algorithm for each trajectory is sum of multiplication of action probability and cumulative sum of discounted rewards. Gradient of loss taken by torch automatic differentiation.

## State normalization
Every component of state, before give it to neural network mapped to \[0;1\] .

## Graphs
Mean reward with correspoding standard deviations evaluated after every 20 batches on 30 trajectories is in **mean_reward.png**.
Trajectories in phase space after some batches are in **./trajectories**. Some representation of policy in phase space in **./phase_space**. 

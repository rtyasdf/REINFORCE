# Description

Each image corresponds to 30 sampled from policies trajectories after 200, 400, 600, 800, 1000 and 1200 batches. 
In phase space main goal can be formulated as *reach an position of 0.5 and higher for less than 200 steps*.

It can be seen that after 200 batches agent learn something from environment -- some trajectories get to leftmost position, from where they can get boost in later episodes.
But still no trajectory can get reward more than -200 and not so much of phase space is explored.

After 400 batches policy learned that it's more beneficial to get to leftmost position, where environment will correct velocity, thus it learns to jump in phase space from one trajectory to the other.
In later episodes this skill will be actively used. Smoothness of trajectories after 1200 batches can be a proof that policy mostly learned dynamics of the system. 

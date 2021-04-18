# Description

Here is actions from phase space, which policy takes after 200, 400, 600, 800, 1000 and 1200 batches. X-axis corresponds to position, Y-axis to velocity. 
Each point on image takes exactly one color: blue -- accelerate forward, red -- accelerate backward and green -- no acceleration.

Because in REINFORCE algorithm policy is stochastic it's not guaranteed that we'll make the same images with the same policy. 
However it's possible to say that after 200 batches policy still hardly understands what action take in most situation -- some amount of green points and mix of red/blue point all over the space.
For 400 bathes and more it's not true anymore -- we clearly can see part space where policy mostly will take acceleration forward(left upper corner) and acceleration backward.
Confidence of the policy here came from the fact that around 320 batches it for the first time reaches flag for less than 200 steps.

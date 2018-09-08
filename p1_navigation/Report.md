# Report: Project 1 - Navigation 
## Introduction 

In this report we present a DeepRL agent solving a variation of [Unity Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) task.
All the available code is contained in this directory. Just run follow the instructions in [README.md](https://github.com/vblagoje/deep-reinforcement-learning/blob/master/p1_navigation/README.md) 

## Solution

The solution involves adapting a vanilla Deep Q-Learning Network we've studied in [Lunar lander](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution) task. 
The adaptation involves adjusting the main agent training code that used Open AI gym environment to Unity ML-agent toolkit environment. 
The neural network is essentially a sequential container of three linear layers with rectified linear unit layers between the first and second, 
and second and third linear layers.
 
This neural network was able to solve the modified banana task in around 400-500 episodes consistently (460 episodes in the example run, see Navigation.ipynb).

There were no changes to the hyper-parameters from the DQN Lunar lander lesson. The decay factor (gamma) was 0.99, the network was trained using 
batch size of 64 samples, the learning rate was unchanged at 0.0005. Target network was updated every four steps. Epsilon starting value was 1, 
the epsilon decay was 0.995, while the minimum possible epsilon value was set to 0.01.  

One can visualize the trained agent's hunt for bananas at https://www.youtube.com/watch?v=1uPnUyaaRNw

## Ideas for future work
+ During the actual use of the trained agent, for some reason the agent gets "confused" if it does not see any yellow bananas in the visual field and it starts to jitter. It would be great to introduce some sort of random search that is less jittery. Not sure how to do this actually, but will investigate.
+ Use other variations of DQN agent, ultimately finishing with Rainbow algorithm to see how fast it would solve the task :-) Perhaps the training plot would be smoother than the plain vanilla DQN as well.

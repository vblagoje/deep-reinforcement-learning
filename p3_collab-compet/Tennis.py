
# coding: utf-8

# # Collaboration and Competition
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import namedtuple, deque
from ddpg_agent import Agent

env = UnityEnvironment(file_name="Tennis.app")


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Traing the agent
# 

# In[5]:


BATCH_SIZE = 128        # minibatch size
BUFFER_SIZE = int(1e6)  # replay buffer size
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
TAU = 6e-2              # for soft update of target parameters


# In[6]:


agent_0 = Agent(0, state_size, action_size, gamma=GAMMA, tau=TAU, replay_buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC)
agent_1 = Agent(1, state_size, action_size, gamma=GAMMA, tau=TAU, replay_buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC)


# In[7]:


def act (states, eps):
    # each agent selects an action
    action_0 = agent_0.act(states, eps)           
    action_1 = agent_1.act(states, eps)
        
    # combine actions and...
    actions = np.concatenate((action_0, action_1), axis=0) 
    actions = np.reshape(actions, (1, action_size*num_agents))
    return actions

def step (states, actions, rewards, next_states, done):
    # let agents step
    agent_0.step(states, actions, rewards[0], next_states, done) 
    agent_1.step(states, actions, rewards[1], next_states, done) 
    

def train(n_episodes=2000, eps=5, eps_end=0.01, eps_decay=0.997, solve_score=0.5): 
    all_scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations
        states = np.reshape(states, (1, state_size*num_agents))
        agent_0.reset()
        agent_1.reset()
        scores = np.zeros(num_agents)
        while True:

            actions = act(states, eps)

            # send them to environment
            env_info = env.step(actions)[brain_name]

            # merge next states into one state space
            next_states = np.reshape(env_info.vector_observations, (1, state_size*2))     

            rewards = env_info.rewards                         
            done = env_info.local_done    

            step(states, actions, rewards, next_states, done)

            scores += rewards                                  
            states = next_states                               

            if np.any(done):                                  
                break

        # for each episode
        eps = max(eps_end, eps*eps_decay)
        scores_window.append(np.max(scores))
        all_scores.append(np.max(scores))

        if i_episode % 10 == 0:
            print('Episode {}\tAverage Reward: {:.3f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >=solve_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_0.pth')
            torch.save(agent_0.critic_local.state_dict(), 'checkpoint_critic_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_1.pth')
            torch.save(agent_1.critic_local.state_dict(), 'checkpoint_critic_1.pth')
            break

    return all_scores    


# In[ ]:


scores = train(solve_score=1.0);



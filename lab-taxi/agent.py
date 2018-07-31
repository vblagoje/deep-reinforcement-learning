import numpy as np
import random
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, alpha=0.25, gamma=0.99, epsilon=0.001):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def choose_eps_greedy_action(self, state):
        if random.random() > self.epsilon and state in self.Q:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.choose_eps_greedy_action(state)

    def expected_sarsa_step(self, state, action, reward, next_state, done):
        if not done:
            action_probs = np.ones(self.nA) * self.epsilon / self.nA
            action_probs[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA)

            alt_estimate = reward + self.gamma * np.dot(self.Q[next_state], action_probs)
            self.Q[state][action] += self.alpha * (alt_estimate - self.Q[state][action])

    def q_learning_step(self, state, action, reward, next_state, done):
        if not done:
            alt_estimate = reward + self.gamma * np.max(self.Q[next_state])
            self.Q[state][action] += self.alpha * (alt_estimate - self.Q[state][action])

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.q_learning_step(state, action, reward, next_state, done)
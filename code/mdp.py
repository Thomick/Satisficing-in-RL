import numpy as np
import matplotlib.pyplot as plt

import sys
from gym import Env, spaces


class MDP(Env):

    def __init__(self, nb_state, nb_action, prob_dist, reward_dist, init_state_dist):
        self.prob_dist = prob_dist
        self.reward_dist = reward_dist
        self.init_state_dist = init_state_dist
        self.nb_state = nb_state
        self.nb_action = nb_action

        self.states = range(0,self.nb_state)
        self.actions = range(0,self.nb_action)

        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(self.nb_action)
        self.observation_space = spaces.Discrete(self.nb_state)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return categorical_sample(self.init_state_dist)

    def step(self, a):
        transitions = self.prob_dist[self.s][a]
        rewarddis = self.reward_dist[self.s][a]
        i = categorical_sample([t[0] for t in transitions])
        p, s, d= transitions[i]
        r =  clip(rewarddis.sample(), self.reward_range)
        self.s = s
        return (s, r, d, "")

def clip(x,range):
    return max(min(x,range[1]),range[0])

def categorical_sample(prob_dist):
    prob_dist = np.array(prob_dist)
    cumprob = np.cumsum(prob_n)
    return (cumprob > np.random.rand()).argmax()
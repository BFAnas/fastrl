import numpy as np
from numpy import *
from numpy.random import normal
import random as rnd
from scipy.special import softmax

rnd.seed(0)


def e_greedy_action_selection(values, eps=0.1):
    if rnd.random() > eps:
        a = argmax(values)
    else:
        n = values.shape[0]
        # selects a random action based on a uniform distribution
        a = rnd.randint(0, n - 1)

    return a, values[a]


def softmax_action_selection(values, tau=1):
    probabilities = softmax(values / tau)
    a = int(np.random.choice(a=np.arange(values.shape[0], dtype=np.int), size=1, replace=False, p=probabilities))

    return a, values[a]


class EpsilonGreedyActionSelection:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.parent = None

    def __call__(self, s):
        return self.select_action(s)

    def select_action(self, s):
        # selects an action using Epsilon-greedy strategy
        # s: the current state
        v = self.parent.Q(s)

        if rnd.random() > self.epsilon:
            a = argmax(v)
        else:
            # selects a random action based on a uniform distribution
            a = rnd.randint(0, self.parent.n_actions - 1)

        return a, v[a]


class SoftmaxActionSelection:
    def __init__(self, tau=1.0):
        self.tau = tau
        self.parent = None

    def __call__(self, s):
        return self.select_action(s)

    def select_action(self, s):
        # selects an action using a kind of soft-max strategy
        # s: the current state
        # this thend to the greedy action when tau tends to zero.

        v = array(self.parent.Q(s))
        probabilities = softmax(v / self.tau)
        a = int(np.random.choice(a=np.arange(v.shape[0], dtype=np.int), size=1, replace=False, p=probabilities))

        return a, v[a]

class ContinousActionSelection:
    def __init__(self, Q):
        self.QFaiss = Q

    def __call__(self, s):
        return self.select_action(s)

    def select_action(self, s):
        knns = self.QFaiss.get_knn_set(s)
        v = self.QFaiss.Q[knns]
        states = [self.QFaiss.all_states[i] for i in knns]
        s_actions = [list(self.QFaiss.get_state_knn_actions(s)) for s in states]
        ac = self.QFaiss.ac
        ai = np.argmax(v, axis=1)
        best_ai = [a[i] for a, i in zip(s_actions, ai)]
        best_ac = [self.QFaiss.all_actions[i] for i in best_ai]
        continous_action = np.sum([i*j for i, j in zip(best_ac, ac)], axis=0)
        return continous_action

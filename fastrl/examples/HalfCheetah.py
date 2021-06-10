from fastrl.algorithms.FARLBasicGYM import FARLBase
import gym
from fastrl.valuefunctions.kNNFaiss import kNNQFaiss
from fastrl.valuefunctions.kNNFaissOffline_ContinousActions import kNNQFaissOfflineContinousActions
import numpy as np
from fastrl.actionselection.ActionSelection import ContinousActionSelection
import pickle
import time
from d3rlpy.datasets import get_pybullet
from d3rlpy.iterators import RandomIterator


def HalfCheetahExperiment(epochs=100, nk=1, evaluation_interval=10):
    print()
    print('===================================================================')
    print('           INIT EXPERIMENT', 'k=' + str(nk + 1))

    # results of the experiment
    x = list(range(1, epochs + 1))
    y = []

    # Build the Environment
    dataset, Env = get_pybullet('halfcheetah-bullet-medium-v0')

    # Build a function approximator
    nk = 10
    Q = kNNQFaissOfflineContinousActions(dataset, action_low=Env.action_space.low, action_high=Env.action_space.high, low=np.clip(Env.observation_space.low, -5, 5), high=np.clip(Env.observation_space.high, -5, 5), k=nk + 1, alpha=5, lm=0)

    # Get the Action Selector, epsilon = 1 means no exploration
    As = ContinousActionSelection(Q)

    # Build the Agent
    MC = FARLBase(Q, Env, As, gamma=0.999)
    MC.Environment.graphs = True
    render = False

    episodes = dataset.episodes
    iterator = RandomIterator(episodes, epochs, 100)

    for i in range(epochs):
        t1 = time.perf_counter()
        batch = next(iterator)
        transitions = batch.transitions
        MC.offline_learning_continous(transitions)
        # Q.reset_traces()
        # result = MC.q_learning_episode(1000)
        t2 = time.perf_counter() - t1
        Q.alpha *= 0.995

        if i%evaluation_interval == 0:
            result = MC.online_evaluation()
            print('Episode', i, 'Episode reward:', result[0], ' Steps:', result[1], 'time:', t2, 'alpha:', Q.alpha, 'epsilon:', 1)
            y.append(result[1])

    return x, y, nk


if __name__ == '__main__':
    HalfCheetahExperiment(epochs=5000, nk=15, evaluation_interval=100)

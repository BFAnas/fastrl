#!/usr/bin/env python3


import numpy as np
from fastrl.valuefunctions.FAInterface import FARL
import faiss
from scipy.special import softmax

class kNNQFaissOfflineContinousActions(FARL):

    def __init__(self, dataset, low, high, action_low, action_high, k=1, alpha=0.3, lm=0.95, nbins=3, ak=3):

        self.episodes = dataset.episodes
        self.all_states = np.concatenate([episode.observations for episode in self.episodes])
        self.all_actions = np.concatenate([episode.actions for episode in self.episodes])
        self.states_actions_dict = dict(zip([str(o.tolist()) for o in self.all_states], self.all_actions))
        self.num_states = len(self.all_states)
        self.action_low = action_low
        self.action_high = action_high
        self.nbins = nbins
        self.action_ndim = len(self.episodes[0].actions[0])
        # Action k param
        self.ak = ak

        print("Constructiong action grid and finding nearest neighbours of dataset actions in the grid")
        distances, knns = self.actions_knns()
        keys = [str(a.tolist()) for a in self.all_actions]
        values = zip(distances, knns)
        self.actions_dict = dict(zip(keys, values))
        actions_set = self.actions_set(knns)
        self.action_size = len(actions_set)


        # Initialize Q values for all state-action paires
        self.Q = np.zeros((self.num_states, k*ak), dtype=np.float32) + -100.0

        # Observation dimension
        self.dimension = int(low.shape[0])
        self.lbounds = low
        self.ubounds = high
        self.cl = np.concatenate([episode.observations for episode in self.episodes], axis=0).astype(np.float32)

        self.k = k
        self.shape = self.cl.shape

        self.ac = []

        self.knn = []
        self.alpha = alpha
        self.lm = lm  # good 0.95
        self.last_state = np.zeros((1, self.shape[1]))

        self.lbounds = np.array(self.lbounds)
        self.ubounds = np.array(self.ubounds)

        self.cl = np.array(self.rescale_inputs(self.cl))

        print("building value function memory")
        res = faiss.StandardGpuResources()  # use a single GPU
        nlist = 100
        quantizer = faiss.IndexFlatL2(self.dimension)  # the other index
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        assert not self.index.is_trained
        self.index.train(self.cl)
        assert self.index.is_trained

        self.index.add(self.cl)
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        print("value function memory done...")

        # self.index.nprobe = 10

    def actions_knns(self):
        nelems = [self.nbins] * self.action_ndim
        ndlinspace = self.action_ndlinspace(nelems).astype(np.float32)
        # Faiss serach
        index = faiss.IndexFlatL2(self.action_ndim)
        index.add(ndlinspace)
        distances, knns = index.search(self.all_actions, self.ak)
        return distances, knns

    def actions_set(self, knns):
        knns = knns.reshape(-1, knns.shape[-1])
        knns = tuple([tuple(n) for n in knns])
        return set(knns)

    def actualize(self):
        self.index.add(x=self.cl)

    def action_ndlinspace(self, nelems):
        x = np.indices(nelems).T.reshape(-1, len(nelems)) + 1.0
        from_b = np.array(nelems, np.float32)
        y = self.action_low + (((x - 1) / (from_b - 1)) * (self.action_high - self.action_low))
        return y

    def random_space(self, npoints):
        d = []
        for l, h in zip(self.lbounds, self.ubounds):
            d.append(np.random.uniform(l, h, (npoints, 1)))

        return np.concatenate(d, 1)

    def load(self, str_filename):
        self.Q = np.load(str_filename)

    def save(self, str_filename):
        np.save(str_filename, self.Q)

    def reset_traces(self):
        self.e *= 0.0
        # self.actualize()

    def rescale_inputs(self, s):
        return self.scale_value(np.array(s), self.lbounds, self.ubounds, -1.0, 1.0)

    def scale_value(self, x, from_a, from_b, to_a, to_b):
        return to_a + (((x - from_a) / (from_b - from_a)) * (to_b - to_a))

    def get_knn_set(self, s):

        if self.last_state is not None:

            if np.allclose(s, self.last_state) and self.knn != []:
                return self.knn

        self.last_state = s
        state = self.rescale_inputs(s)

        d, self.knn = self.index.search(x=np.array([state]).astype(np.float32), k=self.k)
        d = np.squeeze(d)

        self.knn = np.squeeze(self.knn)

        # self.ac = 1.0 / (1.0 + d)  # calculate the degree of activation
        # self.ac /= sum(self.ac)
        self.ac = softmax(-np.sqrt(d))

        return self.knn

    def calculate_knn_q_values(self, M):
        Q_values = np.dot(np.transpose(self.Q[M]), self.ac)
        return Q_values

    def get_value(self, s, idxs=None):
        """ Return the Q value of state (s) for action (a)
        """
        M = self.get_knn_set(s)

        if idxs is None:
            return self.calculate_knn_q_values(M)

        return self.calculate_knn_q_values(M)[idxs]

    def update(self, s, a, v, gamma=1.0):
        """ update action value for action(a)
        """

        M = self.get_knn_set(s)
        # Distances and knns of action in the grid of the action space
        distances, neighbors = self.actions_dict[str(a.tolist())]
        s_knn_actions = self.get_state_knn_actions(s)
        idxs = [self.action_idx(n, s_knn_actions) for n in neighbors]
        a_probs = np.array(distances)/sum(distances)

        td_error = v - self.get_value(s, idxs)
        td_error = np.stack([td_error]*self.k)
        as_prob = np.array([i*j for i in self.ac for j in a_probs]).reshape(self.k, self.ak)
        self.Q[M][:, idxs] += self.alpha * td_error * as_prob

    def has_population(self):
        return True

    def get_population(self):
        pop = self.scale_value(self.cl, -1.0, 1.0, self.lbounds, self.ubounds)
        for i in range(self.shape[0]):
            yield pop[i]

    def get_action_knn_set(self, a):
        return self.actions_dict[str(a.tolist())][1]

    def get_state_knn_actions(self, s):
        knn = self.get_knn_set(s)
        actions = [self.all_actions[n] for n in knn]
        knn_actions = np.concatenate([self.get_action_knn_set(a) for a in actions])
        knn_actions = np.sort(knn_actions)
        return set(tuple(knn_actions))

    def action_idx(self, a, aset):
        return list(aset).index(a)

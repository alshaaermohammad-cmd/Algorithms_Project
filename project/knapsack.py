import numpy as np


class KnapsackProblem:
    def __init__(self, values, weights, capacity, name=""):
        self.values = np.array(values, dtype=np.int64)
        self.weights = np.array(weights, dtype=np.int64)
        self.capacity = int(capacity)
        self.n = len(values)
        self.name = name

        if self.n == 0 or len(weights) != self.n:
            raise ValueError('Values and weights must have the same nonzero length')
        if np.any(self.weights <= 0) or np.any(self.values < 0) or self.capacity < 0:
            raise ValueError('Weights must be positive; values and capacity must be nonnegative')

        self.ratio = self.values / self.weights
        self.order_desc = np.lexsort((np.arange(self.n), -self.ratio))
        self.order_asc = self.order_desc[::-1]
        self._w_list = self.weights.tolist()
        self._desc_list = self.order_desc.tolist()
        self._asc_list = self.order_asc.tolist()

    def fitness(self, x):
        w = int(np.dot(x, self.weights))
        if w > self.capacity:
            return 0
        return int(np.dot(x, self.values))

    def total_weight(self, x):
        return int(np.dot(x, self.weights))

    def repair(self, x):
        xl = x.tolist()
        wl = self._w_list
        w = int(np.dot(x, self.weights))

        if w > self.capacity:
            for i in self._asc_list:
                if xl[i] == 1:
                    xl[i] = 0
                    w -= wl[i]
                    if w <= self.capacity:
                        break

        for i in self._desc_list:
            if xl[i] == 0 and w + wl[i] <= self.capacity:
                xl[i] = 1
                w += wl[i]

        return np.array(xl, dtype=np.int8)


DATA_TYPES = ["uncorrelated", "weak", "strong"]


def generate_instance(n, seed, dtype="weak", name=""):
    rng = np.random.default_rng(seed)
    weights = rng.integers(10, 101, size=n)
    if dtype == "uncorrelated":
        values = rng.integers(10, 101, size=n)
    elif dtype == "weak":
        values = np.maximum(weights + rng.integers(-10, 11, size=n), 1)
    elif dtype == "strong":
        values = weights + 10
    else:
        raise ValueError(dtype)
    capacity = int(weights.sum() // 2)
    return KnapsackProblem(values, weights, capacity, name)


def greedy_solution(problem):
    x = np.zeros(problem.n, dtype=np.int8)
    return problem.repair(x)

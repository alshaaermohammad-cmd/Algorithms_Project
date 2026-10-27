import numpy as np


def dp_knapsack(problem):
    n, W = problem.n, problem.capacity
    wts, vals = problem.weights, problem.values

    dp = np.zeros(W + 1, dtype=np.int64)
    keep = np.zeros((n, W + 1), dtype=bool)

    for i in range(n):
        w, v = int(wts[i]), int(vals[i])
        if w > W:
            continue
        candidate = dp[:W + 1 - w] + v
        better = candidate > dp[w:]
        keep[i, w:] = better
        dp[w:] = np.where(better, candidate, dp[w:])

    x = np.zeros(n, dtype=np.int8)
    c = W
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            x[i] = 1
            c -= int(wts[i])

    return x, int(dp[W])

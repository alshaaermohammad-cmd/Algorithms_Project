import numpy as np


def ant_colony(problem, n_ants=50, max_iter=300, alpha=1.0, beta=2.0, rho=0.1,
               tau_min=0.01, tau_max=1.0, seed=None):
    rng = np.random.default_rng(seed)
    n = problem.n
    wts = problem.weights.tolist()
    W = problem.capacity

    tau = np.full(n, tau_max)
    eta = problem.ratio / problem.ratio.max()

    best_x, best_f = None, -1
    history = []

    for it in range(max_iter + 1):
        attract = (tau ** alpha) * (eta ** beta)

        it_best_x, it_best_f = None, -1
        for _ in range(n_ants):
            keys = -np.log(rng.random(n)) / attract
            order = np.lexsort((np.arange(n), keys))

            x = np.zeros(n, dtype=np.int8)
            load = 0
            for i in order.tolist():
                if load + wts[i] <= W:
                    x[i] = 1
                    load += wts[i]

            f = problem.fitness(x)
            if f > it_best_f:
                it_best_f, it_best_x = f, x

        if it_best_f > best_f:
            best_f, best_x = it_best_f, it_best_x.copy()
        history.append(best_f)

        dep = it_best_x if it % 5 == 4 else best_x
        tau = (1 - rho) * tau
        tau[dep == 1] += rho
        tau = np.clip(tau, tau_min, tau_max)

    return best_x, best_f, history

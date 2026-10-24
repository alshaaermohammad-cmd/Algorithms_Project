import numpy as np


def binary_pso(problem, swarm_size=50, max_iter=300, c1=2.0, c2=2.0,
               w_max=0.9, w_min=0.4, v_max=4.0, transfer='v', seed=None):
    rng = np.random.default_rng(seed)
    n = problem.n

    x = rng.integers(0, 2, size=(swarm_size, n)).astype(np.int8)
    x = np.array([problem.repair(p) for p in x])
    v = rng.uniform(-1, 1, size=(swarm_size, n))
    fit = np.array([problem.fitness(p) for p in x])

    pbest = x.copy()
    pbest_f = fit.copy()
    g = int(np.argmax(fit))
    gbest, gbest_f = x[g].copy(), int(fit[g])
    history = [gbest_f]

    for it in range(max_iter):
        w = w_max - (w_max - w_min) * it / max_iter

        r1 = rng.random((swarm_size, n))
        r2 = rng.random((swarm_size, n))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -v_max, v_max)

        if transfer == 's':
            prob = 1.0 / (1.0 + np.exp(-v))
            x = (rng.random((swarm_size, n)) < prob).astype(np.int8)
        else:
            flip = rng.random((swarm_size, n)) < np.abs(np.tanh(v))
            x = np.where(flip, 1 - x, x).astype(np.int8)

        x = np.array([problem.repair(p) for p in x])
        fit = np.array([problem.fitness(p) for p in x])

        improved = fit > pbest_f
        pbest[improved] = x[improved]
        pbest_f[improved] = fit[improved]

        g = int(np.argmax(pbest_f))
        if pbest_f[g] > gbest_f:
            gbest_f, gbest = int(pbest_f[g]), pbest[g].copy()
        history.append(gbest_f)

    return gbest, gbest_f, history

import numpy as np


def genetic_algorithm(problem, pop_size=50, max_iter=300, pc=0.9, pm=None,
                      tournament_k=3, elite=2, seed=None):
    rng = np.random.default_rng(seed)
    n = problem.n
    if pm is None:
        pm = 1.0 / n

    pop = rng.integers(0, 2, size=(pop_size, n)).astype(np.int8)
    pop = np.array([problem.repair(ind) for ind in pop])
    fit = np.array([problem.fitness(ind) for ind in pop])

    best_idx = int(np.argmax(fit))
    best_x, best_f = pop[best_idx].copy(), int(fit[best_idx])
    history = [best_f]

    for _ in range(max_iter):
        elite_idx = np.lexsort((np.arange(pop_size), -fit))[:elite]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < pop_size:
            p1 = pop[tournament(fit, tournament_k, rng)]
            p2 = pop[tournament(fit, tournament_k, rng)]

            if rng.random() < pc:
                mask = rng.random(n) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for c in (c1, c2):
                flip = rng.random(n) < pm
                c[flip] = 1 - c[flip]
                new_pop.append(problem.repair(c))

        pop = np.array(new_pop[:pop_size])
        fit = np.array([problem.fitness(ind) for ind in pop])

        i = int(np.argmax(fit))
        if fit[i] > best_f:
            best_f, best_x = int(fit[i]), pop[i].copy()
        history.append(best_f)

    return best_x, best_f, history


def tournament(fit, k, rng):
    idx = rng.integers(0, len(fit), size=k)
    return idx[np.argmax(fit[idx])]

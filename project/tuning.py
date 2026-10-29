import csv
import json
import os
from pathlib import Path

import numpy as np

from knapsack import generate_instance, DATA_TYPES
from algorithms.dp import dp_knapsack
from algorithms.ga import genetic_algorithm
from algorithms.pso import binary_pso
from algorithms.aco import ant_colony

RUNS = 5
FUNCS = {"GA": genetic_algorithm, "PSO": binary_pso, "ACO": ant_colony}

GRID = {
    "GA": [{"pm_factor": 1, "tournament_k": 3}, {"pm_factor": 3, "tournament_k": 3},
           {"pm_factor": 1, "tournament_k": 2}],
    "PSO": [{"transfer": "s", "v_max": 4.0}, {"transfer": "v", "v_max": 4.0},
            {"transfer": "v", "v_max": 6.0}],
    "ACO": [{"beta": 2.0, "rho": 0.1}, {"beta": 5.0, "rho": 0.1}, {"beta": 10.0, "rho": 0.1},
            {"beta": 10.0, "rho": 0.05}],
}


def run(algo, params, problem, seed):
    kw = dict(params)
    if "pm_factor" in kw:
        kw["pm"] = kw.pop("pm_factor") / problem.n
    return FUNCS[algo](problem, seed=seed, **kw)[1]


def main():
    output = Path(__file__).resolve().parent / 'results'
    output.mkdir(exist_ok=True)
    problems = []
    for k, (n, dt) in enumerate([(n, dt) for n in (100, 500) for dt in DATA_TYPES]):
        p = generate_instance(n, 1000 + k, dt)
        problems.append((p, dp_knapsack(p)[1]))

    rows, best = [], {}
    for algo, configs in GRID.items():
        for params in configs:
            gaps = [(opt - run(algo, params, p, s)) / opt * 100
                    for p, opt in problems for s in range(RUNS)]
            g = float(np.mean(gaps))
            rows.append({"algorithm": algo, "params": json.dumps(params), "gap_mean_%": round(g, 4)})
            print(algo, params, f"gap={g:.4f}%", flush=True)
            if algo not in best or g < best[algo][1]:
                best[algo] = (params, g)

    with open(output / 'tuning.csv', "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(output / 'best_params.json', "w", encoding="utf-8") as fh:
        json.dump({a: v[0] for a, v in best.items()}, fh, indent=2)
    print("best:", {a: v[0] for a, v in best.items()})


if __name__ == "__main__":
    main()

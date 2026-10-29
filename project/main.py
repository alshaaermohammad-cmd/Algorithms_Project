import argparse
import csv
import json
import os
import statistics
import time
import tracemalloc
import platform
import sys
import hashlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from knapsack import generate_instance, greedy_solution, DATA_TYPES
from algorithms.dp import dp_knapsack
from algorithms.ga import genetic_algorithm
from algorithms.pso import binary_pso
from algorithms.aco import ant_colony

ROOT = Path(__file__).resolve().parent
RES = str(ROOT / 'results')
SIZES = [("Small", 20), ("Medium", 100), ("Large", 500)]
N_INSTANCES = 3
POP, MAX_ITER = 50, 300
TARGET_RATIO = 0.999
FUNCS = {"GA": genetic_algorithm, "PSO": binary_pso, "ACO": ant_colony}
COLORS = {
    "DP": "tab:red",
    "Greedy": "tab:gray",
    "GA": "tab:blue",
    "PSO": "tab:orange",
    "ACO": "tab:green",
}


def load_params():
    path = ROOT / 'results' / 'best_params.json'
    if not os.path.exists(path):
        raise SystemExit("Run tuning.py first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_runner(algo, params):
    def runner(problem, seed):
        kw = dict(params)
        if "pm_factor" in kw:
            kw["pm"] = kw.pop("pm_factor") / problem.n
        size_kw = "swarm_size" if algo == "PSO" else "n_ants" if algo == "ACO" else "pop_size"
        kw[size_kw] = POP
        return FUNCS[algo](problem, max_iter=MAX_ITER, seed=seed, **kw)
    return runner


def peak_memory_kb(func, *args):
    tracemalloc.start()
    func(*args)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1024


def instance_seed(size_idx, type_idx, k):
    return (size_idx + 1) * 100 + type_idx * 10 + k


def first_target_iteration(history, optimum):
    target = TARGET_RATIO * optimum
    for i, f in enumerate(history):
        if f >= target:
            return i
    return None


def timed_deterministic(func, problem, repeats):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(problem)
        times.append(time.perf_counter() - t0)
    return result, statistics.mean(times)


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument('--output', default=str(ROOT / 'results'))
    args = ap.parse_args()
    RES = str(Path(args.output).resolve())
    runs = args.runs
    if runs < 1:
        raise SystemExit("--runs must be >= 1")

    os.makedirs(RES, exist_ok=True)
    params = load_params()
    with open(os.path.join(RES, 'best_params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
    algos = {a: make_runner(a, params[a]) for a in FUNCS}
    run_rows, inst_rows = [], []
    hist = {}
    histories, solutions, problem_rows = [], [], []

    for si, (size, n) in enumerate(SIZES):
        for ti, dt in enumerate(DATA_TYPES):
            for k in range(N_INSTANCES):
                seed_inst = instance_seed(si, ti, k)
                p = generate_instance(n, seed_inst, dt)

                (dp_x, opt), dp_time = timed_deterministic(dp_knapsack, p, runs)
                greedy_x, greedy_time = timed_deterministic(greedy_solution, p, runs)
                greedy_f = p.fitness(greedy_x)
                assert p.total_weight(dp_x) <= p.capacity
                assert p.total_weight(greedy_x) <= p.capacity

                base = {
                    "size": size,
                    "n": n,
                    "type": dt,
                    "instance": k + 1,
                    "instance_seed": seed_inst,
                }
                for item, (weight, value) in enumerate(zip(p.weights, p.values)):
                    problem_rows.append({**base, 'project': item + 1,
                                         'cost': int(weight), 'return': int(value)})
                inst_rows.append({
                    **base,
                    "capacity": p.capacity,
                    "optimum": opt,
                    "dp_time_s": dp_time,
                    "dp_mem_kb": peak_memory_kb(dp_knapsack, p),
                    "greedy_fitness": greedy_f,
                    "greedy_gap_%": (opt - greedy_f) / opt * 100,
                    "greedy_time_s": greedy_time,
                    "greedy_mem_kb": peak_memory_kb(greedy_solution, p),
                })

                for a, run in algos.items():
                    H = []
                    for r in range(runs):
                        t0 = time.perf_counter()
                        x, f, h = run(p, r)
                        elapsed = time.perf_counter() - t0
                        assert p.total_weight(x) <= p.capacity
                        assert f == p.fitness(x) <= opt
                        assert len(h) == MAX_ITER + 1 and h[-1] == f
                        assert np.all(np.diff(h) >= 0)
                        H.append(h)
                        histories.append(np.asarray(h, dtype=np.int64))
                        solutions.append(np.asarray(x, dtype=np.int8))
                        run_rows.append({
                            **base,
                            "algorithm": a,
                            "seed": r,
                            "fitness": f,
                            "optimum": opt,
                            "gap_%": (opt - f) / opt * 100,
                            "time_s": elapsed,
                            "target_reached": first_target_iteration(h, opt) is not None,
                            "target_iteration": first_target_iteration(h, opt),
                        })
                    inst_rows[-1][f"{a}_mem_kb"] = peak_memory_kb(run, p, 0)
                    if k == 0:
                        hist.setdefault((size, dt), {"opt": opt})[a] = np.array(H)

                last = [r for r in run_rows if r["size"] == size and r["type"] == dt
                        and r["instance"] == k + 1]
                print(size, dt, k + 1, "opt", opt,
                      "greedy_gap", round((opt - greedy_f) / opt * 100, 4),
                      {a: round(np.mean([r["gap_%"] for r in last if r["algorithm"] == a]), 4)
                       for a in algos}, flush=True)

    summary = summarize(run_rows, inst_rows, runs)
    save_csv("all_runs.csv", run_rows)
    save_csv("instances.csv", inst_rows)
    save_csv("summary.csv", summary)
    save_csv('problem_data.csv', problem_rows)
    np.savez_compressed(os.path.join(RES, 'histories.npz'), history=np.array(histories))
    np.savez_compressed(os.path.join(RES, 'solutions.npz'),
                        **{f'run_{i}': x for i, x in enumerate(solutions)})
    plot_convergence(hist)
    plot_stability(run_rows)
    plot_time(summary)
    sources = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(ROOT.rglob('*.py'))}
    environment = {'python': platform.python_version(), 'numpy': np.__version__,
                   'matplotlib': matplotlib.__version__, 'system': platform.platform(),
                   'processor': platform.processor(), 'runs_per_instance': runs,
                   'population': POP, 'iterations': MAX_ITER, 'source_sha256': sources}
    with open(os.path.join(RES, 'environment.json'), 'w', encoding='utf-8') as f:
        json.dump(environment, f, indent=2)
    print("done")


def summarize(run_rows, inst_rows, runs):
    out = []
    for size, _ in SIZES:
        for dt in DATA_TYPES:
            ins = [r for r in inst_rows if r["size"] == size and r["type"] == dt]

            out.append({
                "size": size, "type": dt, "algorithm": "DP",
                "gap_mean_%": 0, "gap_std": 0, "gap_worst_%": 0,
                "hits": f"{len(ins)}/{len(ins)}",
                "time_s": np.mean([r["dp_time_s"] for r in ins]),
                "mem_kb": np.mean([r["dp_mem_kb"] for r in ins]),
                "target_hits": "-", "target_iteration_mean": None,
            })

            gg = np.array([r["greedy_gap_%"] for r in ins])
            out.append({
                "size": size, "type": dt, "algorithm": "Greedy",
                "gap_mean_%": gg.mean(), "gap_std": 0, "gap_worst_%": gg.max(),
                "hits": f"{int((gg == 0).sum())}/{len(gg)}",
                "time_s": np.mean([r["greedy_time_s"] for r in ins]),
                "mem_kb": np.mean([r["greedy_mem_kb"] for r in ins]),
                "target_hits": "-", "target_iteration_mean": None,
            })

            for a in FUNCS:
                rr = [r for r in run_rows if r["size"] == size and r["type"] == dt
                      and r["algorithm"] == a]
                g = np.array([r["gap_%"] for r in rr])

                inst_stds = []
                for inst_no in range(1, N_INSTANCES + 1):
                    gi = np.array([r["gap_%"] for r in rr if r["instance"] == inst_no])
                    inst_stds.append(float(gi.std()))

                out.append({
                    "size": size, "type": dt, "algorithm": a,
                    "gap_mean_%": g.mean(),
                    "gap_std": np.mean(inst_stds),
                    "gap_worst_%": g.max(),
                    "hits": f"{int((g == 0).sum())}/{len(g)}",
                    "time_s": np.mean([r["time_s"] for r in rr]),
                    "mem_kb": np.mean([r[f"{a}_mem_kb"] for r in ins]),
                    "target_hits": f"{sum(r['target_reached'] for r in rr)}/{len(rr)}",
                    "target_iteration_mean": (np.mean([r['target_iteration'] for r in rr
                        if r['target_reached']]) if any(r['target_reached'] for r in rr) else None),
                })

    for r in out:
        for key, digits in (("gap_mean_%", 4), ("gap_std", 4), ("gap_worst_%", 4),
                            ("time_s", 12), ("mem_kb", 1)):
            if isinstance(r[key], (float, np.floating)):
                r[key] = round(float(r[key]), digits)
        if isinstance(r['target_iteration_mean'], (float, np.floating)):
            r['target_iteration_mean'] = round(float(r['target_iteration_mean']), 2)
    return out


def plot_convergence(hist):
    for size, n in SIZES:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, dt in zip(axes, DATA_TYPES):
            d = hist[(size, dt)]
            for a in FUNCS:
                ax.plot(d[a].mean(axis=0), color=COLORS[a], lw=1.8, label=a)
            ax.axhline(d["opt"], color=COLORS["DP"], ls="--", lw=1, label="DP optimum")
            lo = min(d[a].mean(axis=0)[0] for a in FUNCS)
            span = max(d["opt"] - lo, 1)
            ax.set_ylim(lo, d["opt"] + span * 0.05)
            ax.set_title(f"{dt} (instance 1)")
            ax.set_xlabel("Iteration")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Best Fitness (mean of runs)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Convergence - {size} (n={n})")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, f"convergence_{size.lower()}.png"), dpi=150)
        plt.close(fig)


def plot_stability(run_rows):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row, (size, n) in enumerate(SIZES):
        for col, dt in enumerate(DATA_TYPES):
            ax = axes[row, col]
            data = [[r["gap_%"] for r in run_rows
                     if r["size"] == size and r["type"] == dt and r["algorithm"] == a]
                    for a in FUNCS]
            ax.boxplot(data, tick_labels=list(FUNCS))
            ax.set_title(f"{size} - {dt}")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("Gap from optimum (%)")
    fig.suptitle("Stability by size and data type")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(RES, "boxplot_gap.png"), dpi=150)
    plt.close(fig)


def plot_time(summary):
    algos = ["DP", "Greedy"] + list(FUNCS)
    x = np.arange(len(SIZES))
    width = 0.16
    plt.figure(figsize=(8, 4))
    for i, a in enumerate(algos):
        t = [np.mean([r["time_s"] for r in summary if r["size"] == s and r["algorithm"] == a])
             for s, _ in SIZES]
        plt.bar(x + (i - 2) * width, t, width, label=a, color=COLORS[a])
    plt.yscale("log")
    plt.xticks(x, [s for s, _ in SIZES])
    plt.ylabel("Mean time per run (s)")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "time_comparison.png"), dpi=150)
    plt.close()


def save_csv(name, rows):
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

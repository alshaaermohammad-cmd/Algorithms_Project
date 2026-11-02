import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from knapsack import generate_instance, greedy_solution
from algorithms.dp import dp_knapsack
import main

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def read_csv(name):
    with (RESULTS / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def verify(rerun=False):
    rows = read_csv('all_runs.csv')
    instances = read_csv('instances.csv')
    saved_summary = read_csv('summary.csv')
    metadata = json.loads((RESULTS / 'environment.json').read_text())
    params = json.loads((RESULTS / 'best_params.json').read_text())
    repeats = metadata['runs_per_instance']
    assert len(rows) == 27 * 3 * repeats
    assert len(instances) == 27
    assert len({(r['size'], r['type'], r['instance'], r['algorithm'], r['seed']) for r in rows}) == len(rows)
    for name, digest in metadata['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    problems = {}
    numeric_instances = []
    for row in instances:
        problem = generate_instance(int(row['n']), int(row['instance_seed']), row['type'])
        choice, optimum = dp_knapsack(problem)
        assert optimum == int(row['optimum']) == problem.fitness(choice)
        assert problem.total_weight(choice) <= problem.capacity
        assert problem.fitness(greedy_solution(problem)) == int(row['greedy_fitness'])
        problems[(row['size'], row['type'], row['instance'])] = problem
        converted = {key: (value if key in ['size', 'type'] else float(value)) for key, value in row.items()}
        converted['instance'] = int(row['instance'])
        numeric_instances.append(converted)

    histories = np.load(RESULTS / 'histories.npz')['history']
    assert histories.shape == (len(rows), main.MAX_ITER + 1)
    numeric_rows = []
    with np.load(RESULTS / 'solutions.npz') as solutions:
        for index, row in enumerate(rows):
            problem = problems[(row['size'], row['type'], row['instance'])]
            choice = solutions[f'run_{index}']
            assert len(choice) == problem.n and np.isin(choice, [0, 1]).all()
            assert problem.total_weight(choice) <= problem.capacity
            score, optimum = int(row['fitness']), int(row['optimum'])
            assert score == problem.fitness(choice) <= optimum
            assert abs(float(row['gap_%']) - 100 * (optimum - score) / optimum) < 1e-9
            history = histories[index]
            assert history[-1] == score and np.all(np.diff(history) >= 0)
            iteration = main.first_target_iteration(history, optimum)
            assert row['target_reached'] == str(iteration is not None)
            assert row['target_iteration'] == (str(iteration) if iteration is not None else '')
            converted = dict(row)
            for key in ['n', 'instance', 'instance_seed', 'seed', 'fitness', 'optimum']:
                converted[key] = int(row[key])
            for key in ['gap_%', 'time_s']:
                converted[key] = float(row[key])
            converted['target_reached'] = iteration is not None
            converted['target_iteration'] = iteration
            numeric_rows.append(converted)
            if rerun and int(row['seed']) == 0:
                _, reproduced, trace = main.make_runner(row['algorithm'], params[row['algorithm']])(problem, 0)
                assert reproduced == score and np.array_equal(trace, history), (row['size'], row['type'], row['instance'], row['algorithm'])

    for saved in saved_summary:
        size, kind, algorithm = saved['size'], saved['type'], saved['algorithm']
        ins = [r for r in numeric_instances if r['size'] == size and r['type'] == kind]
        if algorithm in ['DP', 'Greedy']:
            prefix = 'dp' if algorithm == 'DP' else 'greedy'
            gaps = [0.0] * len(ins) if algorithm == 'DP' else [r['greedy_gap_%'] for r in ins]
            calculated = {'gap_mean_%': round(statistics.mean(gaps), 4), 'gap_std': 0,
                          'gap_worst_%': round(max(gaps), 4),
                          'hits': f'{sum(g == 0 for g in gaps)}/{len(gaps)}',
                          'time_s': round(statistics.mean(r[prefix+'_time_s'] for r in ins), 12),
                          'mem_kb': round(statistics.mean(r[prefix+'_mem_kb'] for r in ins), 1),
                          'target_hits': '-', 'target_iteration_mean': None}
        else:
            selected = [r for r in numeric_rows if r['size'] == size and r['type'] == kind and r['algorithm'] == algorithm]
            gaps = [r['gap_%'] for r in selected]
            reached = [r['target_iteration'] for r in selected if r['target_reached']]
            deviations = [statistics.pstdev(r['gap_%'] for r in selected if r['instance'] == i) for i in range(1, 4)]
            calculated = {'gap_mean_%': round(statistics.mean(gaps), 4),
                          'gap_std': round(statistics.mean(deviations), 4),
                          'gap_worst_%': round(max(gaps), 4),
                          'hits': f'{sum(g == 0 for g in gaps)}/{len(gaps)}',
                          'time_s': round(statistics.mean(r['time_s'] for r in selected), 12),
                          'mem_kb': round(statistics.mean(r[algorithm+'_mem_kb'] for r in ins), 1),
                          'target_hits': f'{len(reached)}/{len(selected)}',
                          'target_iteration_mean': round(statistics.mean(reached), 2) if reached else None}
        for key, value in calculated.items():
            if value is None:
                assert saved[key] == ''
            elif isinstance(value, (int, float, np.floating)):
                assert abs(float(saved[key]) - float(value)) < 1e-8, key
            else:
                assert saved[key] == str(value), key
    print(f'Verified {len(rows)} solutions, histories and summary values.')
    if rerun:
        print('Seed 0 was reproduced for all 81 algorithm-instance combinations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true')
    verify(parser.parse_args().rerun)

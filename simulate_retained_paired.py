"""Execute the frozen, fully synthetic matched observation-loss diagnostic."""
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from selection_risk import estimate, laws
from selection_counts import estimate_counts, constants


def summary(values):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Non-finite diagnostic')
    return dict(mean=float(a.mean()),
                mc_standard_error=float(a.std(ddof=1) / math.sqrt(len(a))))


def losses(result, q, odds):
    return dict(q=float(max(abs(a-b) for a, b in zip(result['q'], q)))**2,
                T=float(max(abs(a-b) for a, b in zip(result['odds'], odds)))**2,
                mean_T=float(result['level']-sum(odds)/len(odds))**2)


def main():
    import argparse
    if sys.flags.optimize:
        raise RuntimeError('Do not disable scientific checks with -O')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    destination = parser.parse_args().output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    seed, reps = 202609274, 200
    rng = np.random.Generator(np.random.PCG64(seed))
    designs = [(F(1), F(2), F(5)), (F(1, 50), F(1, 2), F(40)),
               (F(1), F(1000001, 1000000), F(20))]
    proposals = [[F(1, 2)]*2, [F(1, 10), F(3, 20), F(1, 5), F(1, 4), F(3, 10)]]
    lo, hi, qmin = F(1, 4), F(4), F(1, 20)
    cells = []
    for q in proposals:
        for di, ks in enumerate(designs):
            for delta in [F(0), F(1, 100), F(1, 10), F(1)]:
                for N in [300, 3000, 30000, 300000]:
                    cells.append((q, di, ks, delta, F(3, 2), N, 'main'))
            for center in [F(1, 2), F(7, 2)]:
                for N in [300, 300000]:
                    cells.append((q, di, ks, F(0), center, N, 'flat_level_control'))
    assert len(cells) == 120
    rows = []
    for ci, (q, di, ks, delta, center, N, role) in enumerate(cells):
        s, n = len(q), N//3
        assert N == 3*n
        odds = [center+delta*(F(x, s-1)-F(1, 2)) for x in range(s)]
        assert min(q) >= qmin and sum(q) == 1 and lo <= min(odds) <= max(odds) <= hi
        zs = [sum(a/(1+k*b) for a, b in zip(q, odds)) for k in ks]
        probabilities = laws(q, odds, ks)
        counts = [rng.multinomial(n, [float(p) for p in row], size=reps)
                  for row in probabilities]
        attempts = [n+rng.negative_binomial(n, float(z), size=reps) for z in zs]
        pair_specs = {'retained_first_two': (0, 1), 'retained_endpoints': (0, 2)}
        floors = {name: constants([ks[i] for i in inds], lo, hi, qmin, s)['denominator_floor']
                  for name, inds in pair_specs.items()}
        all_floors = [constants([ks[i], ks[j]], lo, hi, qmin, s)['denominator_floor']
                      for i, j in [(0, 1), (0, 2), (1, 2)]]
        assert floors['retained_endpoints'] == max(all_floors)
        names = ['accepted_only', *pair_specs]
        risks = {name: {key: [] for key in ['q', 'T', 'mean_T']} for name in names}
        diagnostics = {name: dict(zero_cells=0, projected_denominators=0,
                                  clipped_T_endpoints=0) for name in pair_specs}
        for rep in range(reps):
            obs = [[F(int(v), n) for v in row[rep]] for row in counts]
            answers = {'accepted_only': estimate(obs, ks, lo, hi, qmin)}
            for name, inds in pair_specs.items():
                chosen_counts = [[int(v) for v in counts[i][rep]] for i in inds]
                result = estimate_counts(chosen_counts, [int(attempts[i][rep]) for i in inds],
                                         [ks[i] for i in inds], lo, hi, qmin)
                answers[name] = result
                diagnostics[name]['zero_cells'] += sum(v == 0 for row in chosen_counts for v in row)
                diagnostics[name]['projected_denominators'] += sum(d < floors[name] for d in result['denominators'])
                diagnostics[name]['clipped_T_endpoints'] += sum(t in (lo, hi) for t in result['odds'])
            for name, result in answers.items():
                assert sum(result['q']) == 1 and all(0 <= a <= 1 for a in result['q'])
                assert all(lo <= a <= hi for a in result['odds'])
                for key, value in losses(result, q, odds).items():
                    risks[name][key].append(value)
        total_attempts = sum(attempts)
        rows.append(dict(role=role, states=s, design=di, calibrations=list(map(str, ks)),
                         q=list(map(str, q)), odds=list(map(str, odds)), Delta=str(delta),
                         center=str(center), accepted_total=N, accepted_per_regime=n,
                         replicates=reps, acceptance_probabilities=list(map(str, zs)),
                         denominator_floors={k: str(v) for k, v in floors.items()},
                         risks={name: {k: summary(v) for k, v in losses_.items()}
                                for name, losses_ in risks.items()},
                         paired_loss_differences={name: {k: summary(np.asarray(risks[name][k])-
                                                                    np.asarray(risks['accepted_only'][k]))
                                                        for k in ['q', 'T', 'mean_T']}
                                                  for name in pair_specs},
                         diagnostics=diagnostics, total_attempts=summary(total_attempts),
                         expected_total_attempts=str(sum(F(n)/z for z in zs))))
        if (ci+1) % 12 == 0:
            print(json.dumps(dict(completed_cells=ci+1, total_cells=len(cells))), flush=True)
    input_paths = ['DESIGN.md', 'simulate_retained_paired.py', 'selection_counts.py', 'selection_risk.py']
    report = dict(completed_at=datetime.now(timezone.utc).isoformat(),
                  status='EXPLORATORY_SYNTHETIC_RUN_COMPLETE', seed=seed,
                  numpy=np.__version__, bit_generator=type(rng.bit_generator).__name__,
                  domain=dict(m=str(lo), M=str(hi), q_min=str(qmin)), cells=rows,
                  inputs=[dict(path=p, sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
                          for p in input_paths],
                  limits=['Exact fixed-regime sampling factorization; floating-point random generation',
                          'Three estimators see the same stopped experiments; no extra sampling for retention',
                          'Retained endpoint-pair choice uses design bounds only, not observed outcomes',
                          'Finite risk estimates are neither proof nor empirical findings nor an optimality claim',
                          'Physical effort differs across calibration designs despite the same accepted budget'])
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], cells=len(rows), replicates_per_cell=reps)))


if __name__ == '__main__':
    main()

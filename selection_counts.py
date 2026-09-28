"""Exact retained-total inverse for the unselected finite-state experiment.

Inputs are accepted-state counts and the total attempts in each of two fixed
regimes. No rejected labels or source-material inputs are needed. This is a
research prototype, not a verified statistical software package.
"""
from fractions import Fraction as F
from numbers import Integral


def clip(x, lo, hi):
    return min(hi, max(lo, x))


def constants(ks, lo, hi, qmin, states):
    ks = tuple(map(F, ks))
    lo, hi, qmin = F(lo), F(hi), F(qmin)
    if len(ks) != 2 or not 0 < ks[0] < ks[1]:
        raise ValueError("Two ordered strictly positive calibrations required")
    if states < 2 or not 0 < qmin <= F(1, states) or not 0 < lo < hi:
        raise ValueError("Invalid compact finite-support domain")
    k1, k2 = ks
    floor = (k2-k1)*qmin/((1+k1*hi)*(1+k2*hi))
    KT = (2+hi*(k1+k2))/floor
    KQ = 1+k1*hi+k1*KT
    return dict(ks=ks, lo=lo, hi=hi, qmin=qmin, denominator_floor=floor,
                KT=KT, KQ=KQ, Kq=(states+1)*KQ/(states*qmin))


def estimate_masses(masses, ks, lo, hi, qmin):
    """Apply R5/R7 to two subprobability rows, with no data-dependent tolerance."""
    rows = [tuple(map(F, row)) for row in masses]
    if len(rows) != 2 or not rows or len(rows[0]) < 2:
        raise ValueError("Two finite-state mass rows required")
    s = len(rows[0])
    if any(len(row) != s or min(row) < 0 or sum(row) > 1 for row in rows):
        raise ValueError("Each observed mass row must be a nonnegative subprobability vector")
    c = constants(ks, lo, hi, qmin, s)
    k1, k2 = c['ks']
    denominators = [k2*b-k1*a for a,b in zip(*rows)]
    projected = [max(c['denominator_floor'], d) for d in denominators]
    odds = [clip((a-b)/d, c['lo'], c['hi']) for a,b,d in zip(*rows, projected)]
    raw = [a*(1+k1*t) for a,t in zip(rows[0], odds)]
    guarded = [clip(v, c['qmin'], F(1)) for v in raw]
    total = sum(guarded)
    return dict(q=[v/total for v in guarded], odds=odds, level=sum(odds)/s,
                raw_proposal=raw, clipped_proposal=guarded,
                denominators=denominators, projected_denominators=projected)


def estimate_counts(counts, attempts, ks, lo, hi, qmin):
    """Counts/S is (n/S) times the empirical accepted law in each fixed regime."""
    rows = [tuple(row) for row in counts]
    totals = tuple(attempts)
    if len(rows) != 2 or len(totals) != 2:
        raise ValueError("Two count rows and two attempt totals required")
    for row, S in zip(rows, totals):
        if not row or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in row):
            raise ValueError("Accepted counts must be nonnegative integers")
        n = sum(row)
        if isinstance(S, bool) or not isinstance(S, Integral) or not 0 < n <= S:
            raise ValueError("Each total must be an integer at least its positive accepted quota")
    masses = [[F(int(v), int(S)) for v in row] for row,S in zip(rows,totals)]
    result = estimate_masses(masses, ks, lo, hi, qmin)
    result['observed_masses'] = masses
    result['accepted_quotas'] = [sum(row) for row in rows]
    return result

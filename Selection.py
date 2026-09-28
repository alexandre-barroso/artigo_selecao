from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from math import comb, factorial, floor, sqrt
from hashlib import sha256
import ctypes as ct
from numbers import Integral
from pathlib import Path
import os
import random
import re
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import sympy as sp
    import numpy.random._generator as ng
except ImportError:
    raise SystemExit(3)

m, M, q0 = F(1, 4), F(4), F(1, 20)
K = ((F(1), F(2), F(5)), (F(1, 50), F(1, 2), F(40)),
     (F(1), F(1000001, 1000000), F(20)))
Q = ((F(1, 2),) * 2, (F(1, 10), F(3, 20), F(1, 5), F(1, 4), F(3, 10)))
N0 = (300, 3000, 30000, 300000)
Delta0 = (F(0), F(1, 100), F(1, 10), F(1))
J = ((0, 1), (0, 2))
I = (0, 5, 19, 35, 40, 60, 89, 95, 100, 119)
B = 200
seed = 202609274


class C_B(ct.Structure):
    _fields_ = [
        ('h', ct.c_int), ('p', ct.c_double), ('n', ct.c_int64),
        ('r', ct.c_double), ('q', ct.c_double), ('fm', ct.c_double),
        ('m', ct.c_int64), ('p1', ct.c_double), ('xm', ct.c_double),
        ('xl', ct.c_double), ('xr', ct.c_double), ('c', ct.c_double),
        ('ll', ct.c_double), ('lr', ct.c_double), ('p2', ct.c_double),
        ('p3', ct.c_double), ('p4', ct.c_double),
    ]


class G:
    def __init__(self, seed):
        if np.__version__ != '2.4.4' or ct.sizeof(C_B) != 136:
            raise RuntimeError()
        self.g = np.random.Generator(np.random.PCG64(seed))
        self.b = C_B()
        self.ptr = self.g.bit_generator.ctypes.bit_generator
        self.lib = ct.CDLL(ng.__file__)
        self.f = self.lib.random_binomial
        self.f.argtypes = [ct.c_void_p, ct.c_double, ct.c_int64, ct.POINTER(C_B)]
        self.f.restype = ct.c_int64

    def binomial(self, n, p):
        r = min(p, 1.0 - p)
        q = 1.0 - r
        if n * r > 30.0 and (not self.b.h or self.b.n != n or self.b.p != r):
            fm = float(F(n + 1) * F(r))
            m = floor(fm)
            p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5
            xm = m + 0.5
            xl, xr = xm - p1, xm + p1
            c = 0.134 + 20.5 / (15.3 + m)
            a = (fm - xl) / (fm - xl * r)
            ll = a * (1.0 + a / 2.0)
            a = (xr - fm) / (xr * q)
            lr = a * (1.0 + a / 2.0)
            p2 = p1 * (1.0 + 2.0 * c)
            p3 = p2 + c / ll
            p4 = p3 + c / lr
            self.b = C_B(1, r, n, r, q, fm, m, p1, xm, xl, xr, c, ll, lr, p2, p3, p4)
        return self.f(self.ptr, p, n, ct.byref(self.b))

    def multinomial(self, n, p, size):
        if (isinstance(n, bool) or not isinstance(n, Integral) or n < 0
                or not isinstance(size, Integral) or size < 1):
            raise ValueError()
        p = tuple(map(float, p))
        if not p or not all(np.isfinite(p)) or min(p) < 0.0 or max(p) > 1.0:
            raise ValueError()
        if sum(p[:-1]) > 1.0:
            raise ValueError()
        c = np.zeros((size, len(p)), dtype=np.int64)
        for i in range(size):
            d, v = int(n), 1.0
            for j, w in enumerate(p[:-1]):
                c[i, j] = self.binomial(d, w / v)
                d -= int(c[i, j])
                if d <= 0:
                    break
                v -= w
            if d > 0:
                c[i, -1] = d
        return c

    def negative_binomial(self, n, p, size):
        return self.g.negative_binomial(n, p, size=size)


def clip(x, m, M):
    return min(M, max(m, x))


def domain_minus(k, m, M, q0, states):
    k = tuple((F(k) for k in k))
    m, M, q0 = (F(m), F(M), F(q0))
    if len(k) != 3 or not 0 < k[0] < k[1] < k[2]:
        raise ValueError()
    if states < 2 or not 0 < q0 <= F(1, states) or (not 0 < m < M):
        raise ValueError()
    return (k, m, M, q0)


def p_k(q, T, k):
    q, T = (tuple(map(F, q)), tuple(map(F, T)))
    if len(q) != len(T) or len(q) < 2 or sum(q) != 1:
        raise ValueError()
    if min(q) <= 0 or min(T) <= 0:
        raise ValueError()
    result = []
    for raw_k in k:
        k = F(raw_k)
        if k <= 0:
            raise ValueError()
        raw = [p / (1 + k * t) for p, t in zip(q, T)]
        total = sum(raw)
        result.append([v / total for v in raw])
    return result


def pair_minus(y, k, m, M, q0):
    k1, k2, k3 = k
    A = (y[1] - y[0]) / (k2 - k1)
    B = (y[2] - y[1]) / (k3 - k2)
    D, U = (k3 * B - k1 * A, A - B)
    E = (k2 * y[0] - k1 * y[1]) / (k2 - k1)
    b = clip(U / D, m, M) if D else (m + M) / 2
    C = clip(E - k1 * k2 * A * b, q0, 1 / q0)
    a = clip((y[0] * (1 + k1 * b) / C - 1) / k1, m, M)
    h = clip(A * (1 + k1 * b) * (1 + k2 * b) / C, m - M, M - m)
    return dict(y=tuple(y), A=A, B=B, D=D, U=U, E=E, b=b, C=C, a=a, h=h)


def E_minus(observed, k, m, M, q0):
    rows = [tuple(map(F, row)) for row in observed]
    if len(rows) != 3 or not rows:
        raise ValueError()
    states = len(rows[0])
    k, m, M, q0 = domain_minus(k, m, M, q0, states)
    if any((len(row) != states or min(row) < 0 or max(row) > 1 or (sum(row) != 1) for row in rows)):
        raise ValueError()
    eta = q0 * m / M
    guarded = [tuple((max(eta, v) for v in row)) for row in rows]
    pairs = [pair_minus([row[x] / row[0] for row in guarded], k, m, M, q0) for x in range(1, states)]
    c = [F(1)] + [r['C'] for r in pairs]
    q = [v / sum(c) for v in c]
    selected = max(range(len(pairs)), key=lambda i: abs(pairs[i]['D']))
    a = pairs[selected]['a']
    T = [a] + [clip((r['C'] * (1 + k[0] * a) / r['y'][0] - 1) / k[0], m, M) for r in pairs]
    differences = [F(0)] + [r['h'] for r in pairs]
    return dict(q=q, T=T, h=differences, i=selected + 1, P=pairs, p=guarded, tau=sum(T) / states)


def K_minus(k, m, M, q0, states):
    k, m, M, q0 = domain_minus(k, m, M, q0, states)
    k1, k2, k3 = k
    W, eta = (M - m, q0 * m / M)
    LA, LB = (2 / (k2 - k1), 2 / (k3 - k2))
    LD, LU = (k3 * LB + k1 * LA, LA + LB)
    LE, H = ((k2 + k1) / (k2 - k1), (1 + k3 * M) / (k3 - k1))
    P = LU + (M + W) * LD
    KC = LE + k1 * k2 * (M * LA + H * P)
    Y = M / (m * q0)
    A0 = (1 + k1 * M) / (k1 * q0) + Y * (1 + k1 * M) * KC / (k1 * q0 ** 2)
    A1 = Y / q0
    Dmax = W * (k3 - k1) / (q0 * (1 + k1 * m) * (1 + k2 * m) * (1 + k3 * m))
    K0 = Dmax * A0 + A1 * P
    cD = q0 * (k3 - k1) / ((1 + k1 * M) * (1 + k2 * M) * (1 + k3 * M))
    Ka = 2 * (K0 + 2 * LD * W) / cD
    B0 = (1 + k1 * M) * KC / (k1 * eta) + (1 + k1 * M) / (q0 * k1 * eta ** 2)
    B1 = 1 / (q0 * eta)
    KT = max(Ka, W * B0 + B1 * Ka)
    Gmax, LG = ((1 + k1 * M) * (1 + k2 * M), k1 + k2 + 2 * k1 * k2 * M)
    Amax = W / (q0 * (1 + k1 * m) * (1 + k2 * m))
    KH = Gmax * LA / q0 + LG * H * P / q0 + Amax * Gmax * KC / q0 ** 2
    R = 1 / eta + 1 / eta ** 2
    return dict(W=W, eta=eta, LA=LA, LB=LB, LD=LD, LU=LU, LE=LE, H=H, P=P, KC=KC, K0=K0, cD=cD, Ka=Ka, KT=KT, KH=KH, R=R, Kq=states * KC * R)


def K_plus(k, m, M, q0, states):
    k = tuple(map(F, k))
    m, M, q0 = (F(m), F(M), F(q0))
    if len(k) != 2 or not 0 < k[0] < k[1]:
        raise ValueError()
    if states < 2 or not 0 < q0 <= F(1, states) or (not 0 < m < M):
        raise ValueError()
    k1, k2 = k
    floor = (k2 - k1) * q0 / ((1 + k1 * M) * (1 + k2 * M))
    KT = (2 + M * (k1 + k2)) / floor
    KQ = 1 + k1 * M + k1 * KT
    return dict(k=k, m=m, M=M, q0=q0, d0=floor, KT=KT, KQ=KQ, Kq=(states + 1) * KQ / (states * q0))


def E_plus(masses, k, m, M, q0):
    rows = [tuple(map(F, row)) for row in masses]
    if len(rows) != 2 or not rows or len(rows[0]) < 2:
        raise ValueError()
    s = len(rows[0])
    if any((len(row) != s or min(row) < 0 or sum(row) > 1 for row in rows)):
        raise ValueError()
    c = K_plus(k, m, M, q0, s)
    k1, k2 = c['k']
    denominators = [k2 * b - k1 * a for a, b in zip(*rows)]
    projected = [max(c['d0'], d) for d in denominators]
    T = [clip((a - b) / d, c['m'], c['M']) for a, b, d in zip(*rows, projected)]
    raw = [a * (1 + k1 * t) for a, t in zip(rows[0], T)]
    guarded = [clip(v, c['q0'], F(1)) for v in raw]
    total = sum(guarded)
    return dict(q=[v / total for v in guarded], T=T, tau=sum(T) / s, Q0=raw, Q=guarded, d=denominators, dp=projected)


def E_counts(counts, attempts, k, m, M, q0):
    rows = [tuple(row) for row in counts]
    totals = tuple(attempts)
    if len(rows) != 2 or len(totals) != 2:
        raise ValueError()
    for row, S in zip(rows, totals):
        if not row or any((isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in row)):
            raise ValueError()
        n = sum(row)
        if isinstance(S, bool) or not isinstance(S, Integral) or (not 0 < n <= S):
            raise ValueError()
    masses = [[F(int(v), int(S)) for v in row] for row, S in zip(rows, totals)]
    result = E_plus(masses, k, m, M, q0)
    result['r'] = masses
    result['n'] = [sum(row) for row in rows]
    return result


def symbolic_minus():
    C, a, b, k1, k2, k3 = sp.symbols('C a b k1 k2 k3', positive=True)
    y = [C * (1 + k * a) / (1 + k * b) for k in (k1, k2, k3)]
    A, B = ((y[1] - y[0]) / (k2 - k1), (y[2] - y[1]) / (k3 - k2))
    D = k3 * B - k1 * A
    checks = [A * (1 + k1 * b) * (1 + k2 * b) / C - (a - b), (y[0] * (1 + k1 * b) / C - 1) / k1 - a, (C * (1 + k1 * a) / y[0] - 1) / k1 - b, D - C * (a - b) * (k3 - k1) / ((1 + k1 * b) * (1 + k2 * b) * (1 + k3 * b))]
    r, t, delta, k = sp.symbols('r t delta k', positive=True)
    raw1 = r / (1 + k * (t + delta / 2))
    raw0 = (1 - r) / (1 + k * (t - delta / 2))
    checks.append(raw1 / (raw1 + raw0) - (r - r * (1 - r) * k * delta / (1 + k * (t + (1 - 2 * r) * delta / 2))))
    for expression in checks:
        assert sp.factor(expression) == 0
    return len(checks)

def check_minus():
    rng = random.Random(202609272)
    designs = [(F(1), F(2), F(5)), (F(1, 50), F(1, 2), F(40)), (F(1), F(1000001, 1000000), F(20))]
    m, M = (F(1, 4), F(4))
    truth_rows, adversarial, products, flat, zero_D, switched = ([], 0, 0, 0, 0, 0)
    largest_ratios = dict(q=F(0), T=F(0), h=F(0))
    for states in [2, 3, 5, 8]:
        q0 = F(1, 10 * states)
        for design_index, k in enumerate(designs):
            c = K_minus(k, m, M, q0, states)
            for case in range(12):
                weights = [rng.randint(1, 9) for _ in range(states)]
                q = [F(w, sum(weights)) for w in weights]
                assert min(q) >= q0
                if case % 4 == 0:
                    T = [m if case == 0 else M if case == 4 else F(3, 2)] * states
                elif case % 4 == 1:
                    T = [F(3, 2) + F(x, 10 ** 12) for x in range(states)]
                elif case % 4 == 2:
                    T = [F(rng.randint(1, 16), 4) for _ in range(states)]
                else:
                    T = [F(3, 2)] + [m if x % 2 else M for x in range(1, states)]
                Delta = max(T) - min(T)
                ps = p_k(q, T, k)
                fit = E_minus(ps, k, m, M, q0)
                assert fit['q'] == q
                assert fit['h'] == [T[0] - v for v in T]
                if Delta:
                    assert fit['T'] == T
                else:
                    flat += 1
                observed = [ps]
                for eps in [F(1, 10 ** 15), F(1, 100000), F(1, 20)]:
                    obs = []
                    for j, row in enumerate(ps):
                        row = row.copy()
                        source = j % states
                        dest = (j + 1) % states
                        move = min(eps, row[source])
                        row[source] -= move
                        row[dest] += move
                        obs.append(row)
                    observed.append(obs)
                for selected in [0, 1]:
                    observed.append([[F(int(x == selected)) for x in range(states)] for _ in k])
                observed.append([[F(1, states)] * states for _ in k])
                counts = []
                for _ in k:
                    row = [0] * states
                    for _ in range(17):
                        row[rng.randrange(states)] += 1
                    counts.append([F(v, 17) for v in row])
                observed.append(counts)
                truepairs = [pair_minus([row[x] / row[0] for row in ps], k, m, M, q0) for x in range(1, states)]
                Dstar = max((abs(v['D']) for v in truepairs))
                assert Dstar >= c['cD'] * Delta / 2
                for obs in observed:
                    got = E_minus(obs, k, m, M, q0)
                    z = max((abs(x - y) for row, row2 in zip(obs, ps) for x, y in zip(row, row2)))
                    e = max((abs(v['y'][j] - v0['y'][j]) for v, v0 in zip(got['P'], truepairs) for j in range(3)))
                    assert e <= c['R'] * z
                    qerr = max((abs(v - w) for v, w in zip(got['q'], q)))
                    Terr = max((abs(v - w) for v, w in zip(got['T'], T)))
                    herr = max((abs(v - (T[0] - w)) for v, w in zip(got['h'], T)))
                    assert qerr <= c['Kq'] * z
                    assert Delta * Terr <= c['KT'] * e and Terr <= c['W']
                    assert herr <= c['KH'] * e
                    assert Dstar * abs(got['T'][0] - T[0]) <= (c['K0'] + 2 * c['LD'] * c['W']) * e
                    assert sum(got['q']) == 1 and min(got['q']) > 0
                    if got['i'] != fit['i']:
                        switched += 1
                    for x, (g, tr) in enumerate(zip(got['P'], truepairs), 1):
                        b = T[x]
                        assert abs(tr['D']) * abs(g['b'] - b) <= abs(g['U'] - tr['U']) + (abs(b) + c['W']) * abs(g['D'] - tr['D'])
                        assert abs(tr['D']) * abs(g['a'] - T[0]) <= c['K0'] * e
                        products += 1
                        zero_D += int(g['D'] == 0)
                    if z:
                        largest_ratios['q'] = max(largest_ratios['q'], qerr / (c['Kq'] * z))
                    if e:
                        largest_ratios['T'] = max(largest_ratios['T'], Delta * Terr / (c['KT'] * e))
                        largest_ratios['h'] = max(largest_ratios['h'], herr / (c['KH'] * e))
                    adversarial += 1
                truth_rows.append(dict(s=states, j=design_index, i=case, Delta=str(Delta), q=True, h=True, T=bool(Delta)))
    invalid = 0
    for rows, k, lo1, hi1, qmin1 in [([[F(1, 2), F(1, 2)]] * 2, designs[0], m, M, F(1, 10)), ([[F(0), F(0)]] * 3, designs[0], m, M, F(1, 10)), ([[F(1, 2), F(1, 2)]] * 3, [F(1), F(1), F(2)], m, M, F(1, 10)), ([[F(1, 2), F(1, 2)]] * 3, designs[0], m, M, F(3, 5))]:
        try:
            E_minus(rows, k, lo1, hi1, qmin1)
        except ValueError:
            invalid += 1
        else:
            raise AssertionError()
    return (np.array([len(truth_rows), flat, adversarial, products, zero_D, switched, invalid, *map(float, largest_ratios.values())]), np.array([[v['s'], v['j'], v['i'], float(F(v['Delta'])), int(v['q']), int(v['h']), int(v['T'])] for v in truth_rows]))


def check_plus():
    q, T, k1, k2, k, t, u = sp.symbols('q T k1 k2 k t u', positive=True)
    r1, r2 = (q / (1 + k1 * T), q / (1 + k2 * T))
    D = k2 * r2 - k1 * r1
    KL = sp.log((1 + k * u) / (1 + k * t)) + k * t * sp.log(t * (1 + k * u) / (u * (1 + k * t)))
    checks = [sp.simplify(D - (k2 - k1) * q / ((1 + k1 * T) * (1 + k2 * T))) == 0, sp.simplify(r1 - r2 - T * D) == 0, sp.simplify(sp.diff(KL, u) - k * (u - t) / (u * (1 + k * u))) == 0, sp.simplify(KL.subs(u, t)) == 0]
    assert all(checks)
    m, M, q0 = (F(1, 4), F(4), F(1, 20))
    designs = [(F(1), F(2)), (F(1, 50), F(40)), (F(1), F(1000001, 1000000))]
    proposals = [(F(1, 2), F(1, 2)), (q0, 1 - q0), (F(1, 6), F(1, 3), F(1, 2))]
    exact = adversarial = den_floor = zero_cells = 0
    max_T_fraction = max_q_fraction = F(0)
    for k, prob in product(designs, proposals):
        s = len(prob)
        c = K_plus(k, m, M, q0, s)
        for ts in product([m, F(1), F(3, 2), M], repeat=s):
            truth = [[p / (1 + cal * t0) for p, t0 in zip(prob, ts)] for cal in k]
            got = E_plus(truth, k, m, M, q0)
            assert got['q'] == list(prob) and got['T'] == list(ts)
            assert all((d >= c['d0'] for d in got['d']))
            exact += 1
            fixtures = [truth, [[F(0)] * s] * 2, [[F(int(x == 0)) for x in range(s)], [F(int(x == s - 1)) for x in range(s)]], [[F(int(x == s - 1)) for x in range(s)], [F(int(x == 0)) for x in range(s)]], [[F(1, s)] * s] * 2, [[p / F(2) for p in row] for row in truth], [[(1 - F(1, 100)) * p + F(int(x == 0), 100) for x, p in enumerate(row)] for row in truth]]
            for masses in fixtures:
                result = E_plus(masses, k, m, M, q0)
                e = max((abs(a - b) for row, true in zip(masses, truth) for a, b in zip(row, true)))
                lossT = max((abs(a - b) for a, b in zip(result['T'], ts)))
                lossq = max((abs(a - b) for a, b in zip(result['q'], prob)))
                assert lossT <= c['KT'] * e and lossq <= c['Kq'] * e
                assert sum(result['q']) == 1 and min(result['q']) > 0
                assert all((m <= v <= M for v in result['T']))
                if e:
                    max_T_fraction = max(max_T_fraction, lossT / (c['KT'] * e))
                    max_q_fraction = max(max_q_fraction, lossq / (c['Kq'] * e))
                den_floor += sum((a < c['d0'] for a in result['d']))
                zero_cells += sum((v == 0 for row in masses for v in row))
                adversarial += 1
    protocol = 0
    for n, S in product([1, 2, 5, 10], [1, 2, 5, 10, 100, 1000000]):
        if S < n:
            continue
        counts = [[n, 0], [0, n]]
        got = E_counts(counts, [S, S], designs[0], m, M, q0)
        assert got['r'] == [[F(n, S), F(0)], [F(0), F(n, S)]]
        for z in [F(1, 100), F(1, 2), F(99, 100)]:
            assert abs(F(n, S) - z) <= z / F(n) * abs(S - F(n) / z)
        protocol += 1
    witnesses = []
    for s, cal in product([2, 3, 5], [F(1, 1000), F(1), F(1000)]):
        center, h = (F(2), F(1, 10))
        tp = [center + h, center - h] + [center] * (s - 2)
        qp = [(1 + cal * v) / (s * (1 + cal * center)) for v in tp]
        raw = [v / (1 + cal * t0) for v, t0 in zip(qp, tp)]
        assert sum(qp) == 1 and min(qp) >= q0
        assert len(set(tp)) == (3 if s > 2 else 2)
        assert raw == [F(1, s) / (1 + cal * center)] * s
        witnesses.append(dict(s=s, k=str(cal), q=list(map(str, qp)), T=list(map(str, tp))))
    invalids = [([[0, 0], [1, 1]], [1, 2]), ([[1, -1], [1, 1]], [2, 2]), ([[1, 1], [1, 1]], [1, 2]), ([[1, 1], [1, 1]], [2.0, 2]), ([[True, 1], [1, 1]], [2, 2]), ([[1, 1, 0], [1, 1]], [2, 2])]
    rejected = 0
    for counts, attempts in invalids:
        try:
            E_counts(counts, attempts, designs[0], m, M, q0)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError()
    return np.array([len(checks), exact, adversarial, den_floor, zero_cells, protocol, rejected, len(witnesses), float(max_T_fraction), float(max_q_fraction)])


def clip2(value, low=m, high=M):
    return min(high, max(low, value))

def E2_minus(tables, k):
    eta = q0 * m / M
    guarded = [[max(eta, x) for x in row] for row in tables]
    temporary = []
    for x in range(1, len(tables[0])):
        y = [row[x] / row[0] for row in guarded]
        slope_left = (y[1] - y[0]) / (k[1] - k[0])
        slope_right = (y[2] - y[1]) / (k[2] - k[1])
        den = k[2] * slope_right - k[0] * slope_left
        b = clip2((slope_left - slope_right) / den) if den else (m + M) / 2
        ratio = clip2((k[1] * y[0] - k[0] * y[1]) / (k[1] - k[0]) - k[0] * k[1] * slope_left * b, q0, 1 / q0)
        a = clip2((y[0] * (1 + k[0] * b) / ratio - 1) / k[0])
        temporary.append((den, ratio, a, y[0]))
    picked = sorted(range(len(temporary)), key=lambda j: (-abs(temporary[j][0]), j))[0]
    level0 = temporary[picked][2]
    rawq = [F(1)] + [entry[1] for entry in temporary]
    levels = [level0] + [clip2((entry[1] * (1 + k[0] * level0) / entry[3] - 1) / k[0]) for entry in temporary]
    return ([x / sum(rawq) for x in rawq], levels)

def E2_plus(counts, totals, k, indices):
    i, j = indices
    a, b = (k[i], k[j])
    floor = q0 * (b - a) / ((1 + a * M) * (1 + b * M))
    masses_a = [F(int(v), int(totals[i])) for v in counts[i]]
    masses_b = [F(int(v), int(totals[j])) for v in counts[j]]
    rawden = [b * right - a * left for left, right in zip(masses_a, masses_b)]
    levels = [clip2((left - right) / max(floor, d)) for left, right, d in zip(masses_a, masses_b, rawden)]
    rawq = [clip2(mass * (1 + a * t), q0, F(1)) for mass, t in zip(masses_a, levels)]
    diagnostics = (sum((v == 0 for row in [counts[i], counts[j]] for v in row)), sum((d < floor for d in rawden)), sum((t == m or t == M for t in levels)))
    return (([x / sum(rawq) for x in rawq], levels), diagnostics)

def check_law():
    success = (F(1, 5), F(3, 10))
    z = sum(success)
    failure = 1 - z
    tested = 0
    for n in range(1, 4):
        for failures in range(4):
            length = n + failures
            enumerated = defaultdict(F)
            for outcomes in product(range(3), repeat=length):
                if outcomes[-1] == 2 or sum((x == 2 for x in outcomes)) != failures:
                    continue
                weight = F(1)
                for x in outcomes:
                    weight *= failure if x == 2 else success[x]
                enumerated[outcomes.count(0), outcomes.count(1)] += weight
            for c0 in range(n + 1):
                c1 = n - c0
                multinomial = F(factorial(n), factorial(c0) * factorial(c1))
                factored = comb(length - 1, failures) * z ** n * failure ** failures * multinomial * (success[0] / z) ** c0 * (success[1] / z) ** c1
                assert enumerated[c0, c1] == factored
                tested += 1
    return tested

def mu2(values):
    a = np.asarray(values, dtype=float)
    return np.array([float(np.mean(a)), float(np.std(a, ddof=1) / sqrt(a.size))])

def ell2(answer, q, t):
    qhat, that = answer
    return (float(max((abs(a - b) for a, b in zip(qhat, q))) ** 2), float(max((abs(a - b) for a, b in zip(that, t))) ** 2), float((sum(that) / len(that) - sum(t) / len(t)) ** 2))


def theta():
    cells = []
    for q in Q:
        for j, k in enumerate(K):
            for delta in Delta0:
                for N in N0:
                    cells.append((q, j, k, delta, F(3, 2), N, 0))
            for t in (F(1, 2), F(7, 2)):
                for N in (300, 300000):
                    cells.append((q, j, k, F(0), t, N, 1))
    assert len(cells) == 120
    return cells


def mu(values):
    a = np.asarray(values, dtype=float)
    assert a.ndim == 1 and len(a) > 1 and np.isfinite(a).all()
    return np.array((float(a.mean()), float(a.std(ddof=1) / sqrt(len(a)))))


def ell(result, q, T):
    return (
        float(max(abs(a - b) for a, b in zip(result['q'], q))) ** 2,
        float(max(abs(a - b) for a, b in zip(result['T'], T))) ** 2,
        float(result['tau'] - sum(T) / len(T)) ** 2,
    )


def simulate():
    cells = theta()
    rng = G(seed)
    C = np.zeros((120, B, 3, 5), dtype=np.int64)
    S = np.empty((120, B, 3), dtype=np.int64)
    L = np.empty((120, B, 3, 3))
    R = np.empty((120, 3, 3, 2))
    D = np.empty((120, 2, 3, 2))
    H = np.zeros((120, 2, 3), dtype=np.int64)
    W = np.empty((120, 2))
    Z, d0, E = [], [], []
    for ci, (q, j, k, delta, t, N, role) in enumerate(cells):
        s, n = len(q), N // 3
        T = [t + delta * (F(x, s - 1) - F(1, 2)) for x in range(s)]
        assert 3 * n == N and min(q) >= q0 and sum(q) == 1
        assert m <= min(T) <= max(T) <= M
        z = [sum(a / (1 + kj * b) for a, b in zip(q, T)) for kj in k]
        p = p_k(q, T, k)
        counts = [rng.multinomial(n, [float(v) for v in row], size=B) for row in p]
        totals = [n + rng.negative_binomial(n, float(v), size=B) for v in z]
        C[ci, :, :, :s] = np.stack(counts, axis=1)
        S[ci] = np.stack(totals, axis=1)
        floors = [K_plus([k[i] for i in indices], m, M, q0, s)['d0'] for indices in J]
        assert floors[1] == max(
            K_plus([k[a], k[b]], m, M, q0, s)['d0']
            for a, b in ((0, 1), (0, 2), (1, 2))
        )
        for rep in range(B):
            p_hat = [[F(int(v), n) for v in row[rep]] for row in counts]
            answers = [E_minus(p_hat, k, m, M, q0)]
            for ji, indices in enumerate(J):
                c = [[int(v) for v in counts[i][rep]] for i in indices]
                result = E_counts(c, [int(totals[i][rep]) for i in indices],
                                  [k[i] for i in indices], m, M, q0)
                answers.append(result)
                H[ci, ji] += (
                    sum(v == 0 for row in c for v in row),
                    sum(d < floors[ji] for d in result['d']),
                    sum(v in (m, M) for v in result['T']),
                )
            for a, result in enumerate(answers):
                assert sum(result['q']) == 1 and all(0 <= v <= 1 for v in result['q'])
                assert all(m <= v <= M for v in result['T'])
                L[ci, rep, a] = ell(result, q, T)
        for a, b in product(range(3), repeat=2):
            R[ci, a, b] = mu(L[ci, :, a, b])
        for a, b in product(range(2), range(3)):
            D[ci, a, b] = mu(L[ci, :, a + 1, b] - L[ci, :, 0, b])
        W[ci] = mu(sum(totals))
        Z.append(z)
        d0.append(floors)
        E.append(sum(F(n) / v for v in z))
    return dict(theta=cells, C=C, S=S, L=L, R=R, D=D, H=H, W=W, Z=Z, d0=d0, E=E)


def independent(data):
    designs = ((F(1), F(2), F(5)), (F(1, 50), F(1, 2), F(40)),
               (F(1), F(1000001, 1000000), F(20)))
    proposals = ((F(1, 2),) * 2, (F(1, 10), F(3, 20), F(1, 5), F(1, 4), F(3, 10)))
    cells = []
    for q in proposals:
        for j, k in enumerate(designs):
            for delta in (F(0), F(1, 100), F(1, 10), F(1)):
                for N in (300, 3000, 30000, 300000):
                    cells.append((q, j, k, delta, F(3, 2), N, 0))
            for t in (F(1, 2), F(7, 2)):
                for N in (300, 300000):
                    cells.append((q, j, k, F(0), t, N, 1))
    assert len(cells) == len(data['theta']) == 120
    rng = G(202609274)
    V = []
    z_mc = []
    for ci, cell in enumerate(cells):
        assert cell == data['theta'][ci]
        q, j, k, delta, t, N, role = cell
        n, s = N // 3, len(q)
        T = [t + delta * (F(x, s - 1) - F(1, 2)) for x in range(s)]
        r = [[a / (1 + kj * b) for a, b in zip(q, T)] for kj in k]
        z = [sum(row) for row in r]
        assert z == data['Z'][ci]
        E = sum(F(n) / v for v in z)
        assert E == data['E'][ci]
        floor = lambda i, j: F(1, 20) * (k[j] - k[i]) / ((1 + 4 * k[i]) * (1 + 4 * k[j]))
        assert data['d0'][ci] == [floor(0, 1), floor(0, 2)]
        assert floor(0, 2) == max(floor(i, j) for i, j in ((0, 1), (0, 2), (1, 2)))
        R, D = data['R'][ci], data['D'][ci]
        assert np.isfinite(R).all() and np.isfinite(D).all() and (R >= 0).all()
        assert (R[:, 0, 0] <= 1).all() and (R[:, 1:, 0] <= float((M - m) ** 2)).all()
        for a, b in product(range(2), range(3)):
            assert abs(D[a, b, 0] - (R[a + 1, b, 0] - R[0, b, 0])) < 1e-12
            assert abs(R[a + 1, b, 1] - R[0, b, 1]) <= D[a, b, 1] + 1e-12
            assert D[a, b, 1] <= R[a + 1, b, 1] + R[0, b, 1] + 1e-12
        counts = [rng.multinomial(n, [float(v / sum(row)) for v in row], size=200) for row in r]
        totals = [n + rng.negative_binomial(n, float(v), size=200) for v in z]
        assert np.array_equal(np.stack(counts, axis=1), data['C'][ci, :, :, :s])
        assert np.array_equal(np.stack(totals, axis=1), data['S'][ci])
        assert np.max(np.abs(mu2(sum(totals)) - data['W'][ci])) < 1e-8
        z_mc.append((data['W'][ci, 0] - float(E)) / data['W'][ci, 1])
        if ci not in I:
            continue
        L2 = np.empty((200, 3, 3))
        H2 = np.zeros((2, 3), dtype=np.int64)
        for rep in range(200):
            c = [row[rep] for row in counts]
            S = [row[rep] for row in totals]
            p = [[F(int(v), n) for v in row] for row in c]
            answers = [E2_minus(p, k)]
            for ji, indices in enumerate(((0, 1), (0, 2))):
                answer, h = E2_plus(c, S, k, indices)
                answers.append(answer)
                H2[ji] += h
            for a, answer in enumerate(answers):
                L2[rep, a] = ell2(answer, q, T)
        assert np.array_equal(H2, data['H'][ci])
        error = 0.0
        for a, b in product(range(3), repeat=2):
            error = max(error, float(np.max(np.abs(mu2(L2[:, a, b]) - R[a, b]))))
            if a:
                e = mu2(L2[:, a, b] - L2[:, 0, b]) - D[a - 1, b]
                error = max(error, float(np.max(np.abs(e))))
        assert error < 1e-11
        V.append((ci, s, j, float(delta), float(t), N, 200, error))
    assert len(V) * 200 == 2000
    return np.asarray(V), np.asarray(z_mc)


def h(x, n=None):
    a = np.asarray(x, dtype='<f8').ravel()
    if n is not None:
        a = np.asarray([float(format(float(v), f'.{n}e')) for v in a], dtype='<f8')
    return int.from_bytes(sha256(a.tobytes()).digest(), 'big')


def reference(data):
    P = [seed, B, m, M, q0]
    for i, (q, j, k, delta, t, N, role) in enumerate(data['theta']):
        s = len(q)
        T = [t + delta * (F(x, s - 1) - F(1, 2)) for x in range(s)]
        P.extend((s, j, delta, t, N, role, *k, *q, *T,
                  *data['Z'][i], *data['d0'][i], data['E'][i]))
    H = tuple(h(data[k], 12 if k in ('R', 'D') else None)
              for k in ('R', 'D', 'H', 'W')) + (h(P),)
    H0 = (
        54028229598065927471522970401321108801315376646407190221132819692183969076038,
        9647261507012952566036417717670677497944970079143338428591483175269442442582,
        57717392323114743607045269711345585824567002623758136942868756396026109527608,
        42902564979564729852608798830460139003237533916232932583024927158881450313958,
        104668261966237150709644877786941611617670134135572822447978582915615946841567,
    )
    assert H == H0
    assert np.array_equal(np.sum(data['D'][:, :, :, 0] > 0, axis=0),
                          ((46, 19, 29), (12, 10, 1)))
    return np.frombuffer(b''.join(v.to_bytes(32, 'big') for v in H), dtype=np.uint8).reshape(5, 32)


def lean(mathlib):
    root = Path(__file__).resolve().parent
    mathlib = Path(mathlib).resolve(strict=True)
    pin = '5ed2965256430c3649e86755f9576b54eca72435'
    allowed = ('propext', 'Classical.choice', 'Quot.sound')
    targets = (
        ('SelectionStability', (
            'clip_mem', 'clip_distance', 'clippedRatio_mem', 'interval_distance_bound',
            'projected_residual', 'projected_product_bound', 'firstDiff_identity',
            'denominator_identity', 'numerator_identity', 'slope_denominator_identity',
            'proposal_intercept_identity', 'scalar_cancellation_bound',
            'affine_intercept_error', 'absolute_perturbation', 'selected_pair_bound',
            'calibration_fraction_bound',
        )),
        ('SelectionCounts', (
            'count_den_identity', 'count_num_identity', 'count_den_lower_bound',
            'lower_projection_distance', 'projected_inverse_error', 'count_den_error',
            'retained_level_error', 'retained_level_model_error',
            'geometric_derivative_factor', 'rate_estimate_error',
        )),
    )
    def probe(command):
        p = subprocess.run(command, cwd=mathlib, capture_output=True, text=True,
                           timeout=40, check=True)
        return p.stdout.strip()
    version = probe(['lake', 'env', 'lean', '--version'])
    assert re.search(r'\b4\.34\.0\b', version)
    assert probe(['git', 'rev-parse', 'HEAD']) == pin
    prefix = probe(['lake', 'env', 'lean', '--print-prefix'])
    imports = probe(['lake', 'env', 'printenv', 'LEAN_PATH'])
    A = []
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        env = os.environ.copy()
        env['LEAN_PATH'] = str(out) + os.pathsep + imports
        for module, names in targets:
            source = root / (module + '.lean')
            raw = source.read_bytes()
            body = raw.decode()
            assert not re.search(r'^\s*(?:axiom|opaque|unsafe)\s|\b(?:sorry|admit|native_decide)\b', body, re.M)
            assert re.findall(r'^#print axioms (\w+)\s*$', body, re.M) == list(names)
            staged, olean = out / source.name, out / (module + '.olean')
            staged.write_bytes(raw)
            p = subprocess.run(
                [str(Path(prefix) / 'bin' / 'lean'), '--trust=0', '--root=' + str(out),
                 '-o', str(olean), str(staged)],
                cwd=mathlib, env=env, capture_output=True, text=True, timeout=180,
            )
            text = p.stdout + p.stderr
            axioms = {n: [x.strip() for x in v.split(',') if x.strip()]
                      for n, v in re.findall(r"^'([^'\n]+)'[^\[\]\n]*\s*\[([^]]*)\]", text, re.M)}
            axioms.update({n: [] for n in re.findall(r"^'([^'\n]+)'[^\n:\[\]]*\baxioms\s*$", text, re.M)})
            assert p.returncode == 0 and olean.is_file()
            assert source.read_bytes() == staged.read_bytes() == raw
            assert set(axioms) == {module + '.' + n for n in names}
            assert all(set(v) <= set(allowed) for v in axioms.values())
            A.extend([int(a in axioms[module + '.' + n]) for a in allowed] for n in names)
    assert len(A) == 26
    return np.asarray(A, dtype=np.int64)


def figure(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = data['R']
    H = np.full((17, 9), np.nan)
    rho = np.empty((3, 2, 4, 4))
    for j, c, a, b in product(range(3), range(2), range(4), range(4)):
        i = 60 * c + 20 * j + 4 * a + b
        assert data['theta'][i][-1] == 0
        rho[j, c, a, b] = R[i, 2, 1, 0] / R[i, 0, 1, 0]
        H[6 * j + 1 + a, 5 * c + b] = np.log10(rho[j, c, a, b])
    assert np.isfinite(rho).all() and (rho > 0).all()
    fig = plt.figure(figsize=(8.3, 7.8))
    ax = fig.add_axes((0.08, 0.15, 0.88, 0.79))
    im = ax.imshow(H, vmin=-3, vmax=3, aspect='auto', alpha=0.5)
    for j, c, a, b in product(range(3), range(2), range(4), range(4)):
        ax.text(5 * c + b, 6 * j + 1 + a,
                f'{rho[j, c, a, b]:.2g}'.replace('.', ','),
                ha='center', va='center', fontsize=9)
    names = (r'(1;2;5)', r'(0{,}02;0{,}5;40)', r'(1;1{,}000001;20)')
    for j, c in product(range(3), range(2)):
        ax.text(5 * c + 1.5, 6 * j,
                rf'$s={2 if c == 0 else 5},\quad k={names[j]}$',
                ha='center', va='center', fontsize=10)
    ax.set_xticks((0, 1, 2, 3, 5, 6, 7, 8),
                  [r'$300$', r'$3000$', r'$30000$', r'$300000$'] * 2)
    ax.set_yticks([6 * j + 1 + a for j in range(3) for a in range(4)],
                  [r'$0$', r'$0{,}01$', r'$0{,}1$', r'$1$'] * 3)
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$\Delta$')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.add_axes((0.18, 0.075, 0.68, 0.023))
    fig.colorbar(im, cax=cb, orientation='horizontal', extend='both',
                 label=r'$\log_{10}(\widehat{R}_{T}^{+,(1,3)}/\widehat{R}_{T}^{-})$')
    fig.savefig(out / 'figura1.pdf')
    fig.savefig(out / 'figura1.png', dpi=300)
    plt.close(fig)
    return rho


def save(data, V0, F0, V1, V2, V3, V4, A, out):
    cells = data['theta']
    q = np.zeros((120, 5))
    T = np.zeros((120, 5))
    k = np.empty((120, 3))
    p = np.zeros((120, 3, 5))
    rows = []
    for i, (qx, j, ki, delta, t, N, role) in enumerate(cells):
        s = len(qx)
        tx = [t + delta * (F(x, s - 1) - F(1, 2)) for x in range(s)]
        q[i, :s], T[i, :s], k[i] = qx, tx, ki
        p[i, :, :s] = p_k(qx, tx, ki)
        rows.append((s, j, float(delta), float(t), N, N // 3, B, role))
    table = np.sum(data['D'][:, :, :, 0] > 0, axis=0)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        rho = figure(data, tmp)
        np.savetxt(tmp / 'tabela2.csv', table, delimiter=',', fmt='%d')
        numerical = []
        for i, a, b in product(range(120), range(3), range(3)):
            d = data['D'][i, a - 1, b] if a else (0.0, 0.0)
            numerical.append((i, *rows[i], a, b, *data['R'][i, a, b], *d,
                              float(data['E'][i]), *data['W'][i]))
        np.savetxt(tmp / 'resultados.csv', numerical, delimiter=',', fmt='%.17g')
        np.savez_compressed(
            tmp / 'resultados.npz',
            theta=np.asarray(rows), q=q, T=T, k=k, p=p,
            Z=np.asarray(data['Z'], dtype=float), d0=np.asarray(data['d0'], dtype=float),
            E=np.asarray(data['E'], dtype=float), C=data['C'], S=data['S'],
            L=data['L'], R=data['R'], D=data['D'], H=data['H'], W=data['W'],
            V0=V0, F0=F0, V1=V1, V2=V2, V3=V3, V4=V4, A=A,
            B=table, rho=rho, seed=seed,
            python=np.asarray(sys.version_info[:3]),
            numpy=np.asarray([int(v) for v in np.__version__.split('.')[:3]]),
            sympy=np.asarray([int(v) for v in sp.__version__.split('.')[:3]]),
        )
        out.mkdir(parents=True, exist_ok=True)
        for f in tmp.iterdir():
            destination = out / f.name
            with f.open('rb') as source, tempfile.NamedTemporaryFile(dir=out, delete=False) as target:
                path = Path(target.name)
                target.write(source.read())
            try:
                os.replace(path, destination)
            finally:
                path.unlink(missing_ok=True)


def main():
    if sys.flags.optimize:
        raise SystemExit(2)
    if np.__version__ != '2.4.4' or sp.__version__ != '1.14.0':
        raise SystemExit(3)
    args = sys.argv[1:]
    if len(args) % 2 or any(a not in ('--output', '--mathlib') for a in args[::2]):
        raise SystemExit(2)
    options = dict(zip(args[::2], args[1::2]))
    if len(options) != len(args) // 2:
        raise SystemExit(2)
    out = Path(options.get('--output', 'resultados_selecao')).resolve()
    V0, F0 = check_minus()
    V0 = np.concatenate(([symbolic_minus()], V0))
    V1 = check_plus()
    n = check_law()
    assert n == 36
    data = simulate()
    V2, V3 = independent(data)
    V4 = reference(data)
    A = np.empty((0, 3), dtype=np.int64)
    if '--mathlib' in options:
        A = lean(options['--mathlib'])
    V1 = np.concatenate((V1, [n]))
    save(data, V0, F0, V1, V2, V3, V4, A, out)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        raise SystemExit(1)

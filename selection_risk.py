"""Exact arithmetic prototype for an unselected calibrated-selection result.

Synthetic research, not an empirical model or a certified statistical package.
The three regime rows contain accepted-label proportions. Attempt totals are
not inputs. Fractions preserve the finite arithmetic exactly; converting a float
uses its actual binary value. No source-material code or data is imported.
"""
from fractions import Fraction as F


def clip(x, lo, hi):
    return min(hi, max(lo, x))


def domain(ks, lo, hi, qmin, states):
    ks = tuple(F(k) for k in ks)
    lo, hi, qmin = F(lo), F(hi), F(qmin)
    if len(ks) != 3 or not 0 < ks[0] < ks[1] < ks[2]:
        raise ValueError("Exactly three strictly increasing positive calibrations required")
    if states < 2 or not 0 < qmin <= F(1, states) or not 0 < lo < hi:
        raise ValueError("Invalid fixed finite-support domain")
    return ks, lo, hi, qmin


def laws(q, odds, ks):
    q, odds = tuple(map(F, q)), tuple(map(F, odds))
    if len(q) != len(odds) or len(q) < 2 or sum(q) != 1:
        raise ValueError("Proposal must be a probability vector on the same finite support")
    if min(q) <= 0 or min(odds) <= 0:
        raise ValueError("Strictly positive probabilities and odds required")
    result = []
    for raw_k in ks:
        k = F(raw_k)
        if k <= 0:
            raise ValueError("Calibrations must be positive")
        raw = [p / (1 + k * t) for p, t in zip(q, odds)]
        total = sum(raw)
        result.append([v / total for v in raw])
    return result


def pair(y, ks, lo, hi, qmin):
    k1, k2, k3 = ks
    A = (y[1] - y[0]) / (k2 - k1)
    B = (y[2] - y[1]) / (k3 - k2)
    D, U = k3 * B - k1 * A, A - B
    E = (k2 * y[0] - k1 * y[1]) / (k2 - k1)
    b = clip(U / D, lo, hi) if D else (lo + hi) / 2
    C = clip(E - k1 * k2 * A * b, qmin, 1 / qmin)
    a = clip((y[0] * (1 + k1 * b) / C - 1) / k1, lo, hi)
    h = clip(A * (1 + k1 * b) * (1 + k2 * b) / C, lo - hi, hi - lo)
    return dict(y=tuple(y), A=A, B=B, D=D, U=U, E=E, b=b, C=C, a=a, h=h)


def estimate(observed, ks, lo, hi, qmin):
    """Return proposal, absolute odds and reference differences, using v2 formulas."""
    rows = [tuple(map(F, row)) for row in observed]
    if len(rows) != 3 or not rows:
        raise ValueError("Exactly three accepted-law rows required")
    states = len(rows[0])
    ks, lo, hi, qmin = domain(ks, lo, hi, qmin, states)
    if any(len(row) != states or min(row) < 0 or max(row) > 1 or sum(row) != 1 for row in rows):
        raise ValueError("Each observed row must be a probability vector")
    eta = qmin * lo / hi
    guarded = [tuple(max(eta, v) for v in row) for row in rows]
    pairs = [pair([row[x] / row[0] for row in guarded], ks, lo, hi, qmin)
             for x in range(1, states)]
    c = [F(1)] + [r["C"] for r in pairs]
    q = [v / sum(c) for v in c]
    # Python's first-maximum convention is the deterministic tie rule.
    selected = max(range(len(pairs)), key=lambda i: abs(pairs[i]["D"]))
    a = pairs[selected]["a"]
    odds = [a] + [clip((r["C"] * (1 + ks[0] * a) / r["y"][0] - 1) / ks[0], lo, hi)
                  for r in pairs]
    differences = [F(0)] + [r["h"] for r in pairs]
    return dict(q=q, odds=odds, differences=differences, selected_cell=selected + 1,
                pairs=pairs, guarded=guarded, level=sum(odds) / states)


def constants(ks, lo, hi, qmin, states):
    """Conservative exact constants appearing in the v2 deterministic proof."""
    ks, lo, hi, qmin = domain(ks, lo, hi, qmin, states)
    k1, k2, k3 = ks
    W, eta = hi - lo, qmin * lo / hi
    LA, LB = 2 / (k2 - k1), 2 / (k3 - k2)
    LD, LU = k3 * LB + k1 * LA, LA + LB
    LE, H = (k2 + k1) / (k2 - k1), (1 + k3 * hi) / (k3 - k1)
    P = LU + (hi + W) * LD
    KC = LE + k1 * k2 * (hi * LA + H * P)
    Y = hi / (lo * qmin)
    A0 = (1 + k1 * hi) / (k1 * qmin) + Y * (1 + k1 * hi) * KC / (k1 * qmin**2)
    A1 = Y / qmin
    Dmax = W * (k3 - k1) / (qmin * (1 + k1 * lo) * (1 + k2 * lo) * (1 + k3 * lo))
    K0 = Dmax * A0 + A1 * P
    cD = qmin * (k3 - k1) / ((1 + k1 * hi) * (1 + k2 * hi) * (1 + k3 * hi))
    Ka = 2 * (K0 + 2 * LD * W) / cD
    B0 = (1 + k1 * hi) * KC / (k1 * eta) + (1 + k1 * hi) / (qmin * k1 * eta**2)
    B1 = 1 / (qmin * eta)
    KT = max(Ka, W * B0 + B1 * Ka)
    Gmax, LG = (1 + k1 * hi) * (1 + k2 * hi), k1 + k2 + 2 * k1 * k2 * hi
    Amax = W / (qmin * (1 + k1 * lo) * (1 + k2 * lo))
    KH = Gmax * LA / qmin + LG * H * P / qmin + Amax * Gmax * KC / qmin**2
    R = 1 / eta + 1 / eta**2
    return dict(W=W, eta=eta, LA=LA, LB=LB, LD=LD, LU=LU, LE=LE, H=H, P=P,
                KC=KC, K0=K0, cD=cD, Ka=Ka, KT=KT, KH=KH, R=R, Kq=states * KC * R)

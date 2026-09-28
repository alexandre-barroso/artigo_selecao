"""Symbolic and exact adversarial checks of the retained-total prototype."""
from datetime import datetime, timezone
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import argparse
import json
import platform
import sys
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from selection_counts import constants, estimate_counts, estimate_masses


def main():
    if sys.flags.optimize:
        raise RuntimeError('Do not disable scientific checks with -O')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists()
    q,T,k1,k2,k,t,u = sp.symbols('q T k1 k2 k t u', positive=True)
    r1,r2=q/(1+k1*T),q/(1+k2*T)
    D=k2*r2-k1*r1
    KL=sp.log((1+k*u)/(1+k*t))+k*t*sp.log(t*(1+k*u)/(u*(1+k*t)))
    checks = [sp.simplify(D-(k2-k1)*q/((1+k1*T)*(1+k2*T))) == 0,
              sp.simplify(r1-r2-T*D) == 0,
              sp.simplify(sp.diff(KL,u)-k*(u-t)/(u*(1+k*u))) == 0,
              sp.simplify(KL.subs(u,t)) == 0]
    assert all(checks)
    lo,hi,qmin=F(1,4),F(4),F(1,20)
    designs=[(F(1),F(2)),(F(1,50),F(40)),(F(1),F(1000001,1000000))]
    proposals=[(F(1,2),F(1,2)),(qmin,1-qmin),(F(1,6),F(1,3),F(1,2))]
    exact=adversarial=den_floor=zero_cells=0
    max_T_fraction=max_q_fraction=F(0)
    for ks,prob in product(designs,proposals):
        s=len(prob)
        c=constants(ks,lo,hi,qmin,s)
        for ts in product([lo,F(1),F(3,2),hi],repeat=s):
            truth=[[p/(1+cal*t0) for p,t0 in zip(prob,ts)] for cal in ks]
            got=estimate_masses(truth,ks,lo,hi,qmin)
            assert got['q']==list(prob) and got['odds']==list(ts)
            assert all(d>=c['denominator_floor'] for d in got['denominators'])
            exact+=1
            fixtures=[truth, [[F(0)]*s]*2,
                [[F(int(x==0)) for x in range(s)], [F(int(x==s-1)) for x in range(s)]],
                [[F(int(x==s-1)) for x in range(s)], [F(int(x==0)) for x in range(s)]],
                [[F(1,s)]*s]*2,
                [[p/F(2) for p in row] for row in truth],
                [[(1-F(1,100))*p+F(int(x==0),100) for x,p in enumerate(row)] for row in truth]]
            for masses in fixtures:
                result=estimate_masses(masses,ks,lo,hi,qmin)
                e=max(abs(a-b) for row,true in zip(masses,truth) for a,b in zip(row,true))
                lossT=max(abs(a-b) for a,b in zip(result['odds'],ts))
                lossq=max(abs(a-b) for a,b in zip(result['q'],prob))
                assert lossT<=c['KT']*e and lossq<=c['Kq']*e
                assert sum(result['q'])==1 and min(result['q'])>0
                assert all(lo<=v<=hi for v in result['odds'])
                if e:
                    max_T_fraction=max(max_T_fraction,lossT/(c['KT']*e))
                    max_q_fraction=max(max_q_fraction,lossq/(c['Kq']*e))
                den_floor+=sum(a<c['denominator_floor'] for a in result['denominators'])
                zero_cells+=sum(v==0 for row in masses for v in row)
                adversarial+=1
    protocol=0
    for n,S in product([1,2,5,10],[1,2,5,10,100,1000000]):
        if S<n:
            continue
        counts=[[n,0],[0,n]]
        got=estimate_counts(counts,[S,S],designs[0],lo,hi,qmin)
        assert got['observed_masses']==[[F(n,S),F(0)],[F(0),F(n,S)]]
        for z in [F(1,100),F(1,2),F(99,100)]:
            assert abs(F(n,S)-z)<=z/F(n)*abs(S-F(n)/z)
        protocol+=1
    witnesses=[]
    for s,cal in product([2,3,5],[F(1,1000),F(1),F(1000)]):
        center,h=F(2),F(1,10)
        tp=[center+h,center-h]+[center]*(s-2)
        qp=[(1+cal*v)/(s*(1+cal*center)) for v in tp]
        raw=[v/(1+cal*t0) for v,t0 in zip(qp,tp)]
        assert sum(qp)==1 and min(qp)>=qmin
        assert len(set(tp)) == (3 if s>2 else 2)
        assert raw==[F(1,s)/(1+cal*center)]*s
        witnesses.append(dict(states=s,k=str(cal),q=list(map(str,qp)),T=list(map(str,tp))))
    invalids=[([[0,0],[1,1]],[1,2]),([[1,-1],[1,1]],[2,2]),
              ([[1,1],[1,1]],[1,2]),([[1,1],[1,1]],[2.0,2]),
              ([[True,1],[1,1]],[2,2]),([[1,1,0],[1,1]],[2,2])]
    rejected=0
    for counts,attempts in invalids:
        try:
            estimate_counts(counts,attempts,designs[0],lo,hi,qmin)
        except ValueError:
            rejected+=1
        else:
            raise AssertionError('Invalid protocol accepted')
    paths=['selection_counts.py','check_selection_counts.py']
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),passed=True,python=platform.python_version(),sympy=sp.__version__,
        symbolic_identities=len(checks),population_fixtures=exact,adversarial_fixtures=adversarial,
        projected_denominators=den_floor,zero_cells=zero_cells,protocol_fixtures=protocol,
        rejected_invalid_protocols=rejected,one_regime_nonidentification=witnesses,
        maximum_fraction_of_conservative_T_bound=str(max_T_fraction),maximum_fraction_of_conservative_q_bound=str(max_q_fraction),
        inputs=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in paths],
        limits=['Finite exact checks do not prove statistical risk, adaptive KL or novelty',
                'No empirical observations or simulation used','Population fixtures do not validate sample performance'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','one_regime_nonidentification']}))


if __name__=='__main__':
    main()

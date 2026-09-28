"""Adversarial finite exact checks of the new multistate v2 proof obligations."""
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import random
import sys
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from selection_risk import constants, estimate, laws, pair


def symbolic():
    C, a, b, k1, k2, k3 = sp.symbols("C a b k1 k2 k3", positive=True)
    y = [C * (1 + k*a) / (1 + k*b) for k in (k1, k2, k3)]
    A, B = (y[1]-y[0])/(k2-k1), (y[2]-y[1])/(k3-k2)
    D = k3*B-k1*A
    checks = [A*(1+k1*b)*(1+k2*b)/C - (a-b),
        (y[0]*(1+k1*b)/C-1)/k1-a,
        (C*(1+k1*a)/y[0]-1)/k1-b,
        D-C*(a-b)*(k3-k1)/((1+k1*b)*(1+k2*b)*(1+k3*b))]
    r,t,delta,k = sp.symbols("r t delta k", positive=True)
    raw1=r/(1+k*(t+delta/2))
    raw0=(1-r)/(1+k*(t-delta/2))
    checks.append(raw1/(raw1+raw0)-(r-r*(1-r)*k*delta/(1+k*(t+(1-2*r)*delta/2))))
    for expression in checks:
        assert sp.factor(expression) == 0
    return len(checks)


def run():
    rng = random.Random(202609272)
    designs = [(F(1),F(2),F(5)), (F(1,50),F(1,2),F(40)), (F(1),F(1000001,1000000),F(20))]
    lo, hi = F(1,4), F(4)
    truth_rows, adversarial, products, flat, zero_D, switched = [], 0, 0, 0, 0, 0
    largest_ratios = dict(q=F(0), absolute=F(0), difference=F(0))
    for states in [2,3,5,8]:
        qmin=F(1,10*states)
        for design_index,ks in enumerate(designs):
            c=constants(ks,lo,hi,qmin,states)
            for case in range(12):
                weights=[rng.randint(1,9) for _ in range(states)]
                q=[F(w,sum(weights)) for w in weights]
                assert min(q)>=qmin
                if case%4==0:
                    odds=[lo if case==0 else hi if case==4 else F(3,2)]*states
                elif case%4==1:
                    odds=[F(3,2)+F(x,10**12) for x in range(states)]
                elif case%4==2:
                    odds=[F(rng.randint(1,16),4) for _ in range(states)]
                else:
                    odds=[F(3,2)]+[lo if x%2 else hi for x in range(1,states)]
                Delta=max(odds)-min(odds)
                ps=laws(q,odds,ks)
                fit=estimate(ps,ks,lo,hi,qmin)
                assert fit['q']==q
                assert fit['differences']==[odds[0]-v for v in odds]
                if Delta:
                    assert fit['odds']==odds
                else:
                    flat+=1
                observed=[ps]
                for eps in [F(1,10**15),F(1,100000),F(1,20)]:
                    obs=[]
                    for j,row in enumerate(ps):
                        row=row.copy()
                        source=j%states
                        dest=(j+1)%states
                        move=min(eps,row[source])
                        row[source]-=move
                        row[dest]+=move
                        obs.append(row)
                    observed.append(obs)
                for selected in [0,1]:
                    observed.append([[F(int(x==selected)) for x in range(states)] for _ in ks])
                observed.append([[F(1,states)]*states for _ in ks])
                counts=[]
                for _ in ks:
                    row=[0]*states
                    for _ in range(17):
                        row[rng.randrange(states)]+=1
                    counts.append([F(v,17) for v in row])
                observed.append(counts)
                truepairs=[pair([row[x]/row[0] for row in ps],ks,lo,hi,qmin) for x in range(1,states)]
                Dstar=max(abs(v['D']) for v in truepairs)
                assert Dstar>=c['cD']*Delta/2
                for obs in observed:
                    got=estimate(obs,ks,lo,hi,qmin)
                    z=max(abs(x-y) for row,row2 in zip(obs,ps) for x,y in zip(row,row2))
                    e=max(abs(v['y'][j]-v0['y'][j]) for v,v0 in zip(got['pairs'],truepairs) for j in range(3))
                    assert e<=c['R']*z
                    qerr=max(abs(v-w) for v,w in zip(got['q'],q))
                    Terr=max(abs(v-w) for v,w in zip(got['odds'],odds))
                    herr=max(abs(v-(odds[0]-w)) for v,w in zip(got['differences'],odds))
                    assert qerr<=c['Kq']*z
                    assert Delta*Terr<=c['KT']*e and Terr<=c['W']
                    assert herr<=c['KH']*e
                    assert Dstar*abs(got['odds'][0]-odds[0])<=(c['K0']+2*c['LD']*c['W'])*e
                    assert sum(got['q'])==1 and min(got['q'])>0
                    if got['selected_cell']!=fit['selected_cell']:
                        switched+=1
                    for x,(g,tr) in enumerate(zip(got['pairs'],truepairs),1):
                        b=odds[x]
                        assert abs(tr['D'])*abs(g['b']-b)<=abs(g['U']-tr['U'])+(abs(b)+c['W'])*abs(g['D']-tr['D'])
                        assert abs(tr['D'])*abs(g['a']-odds[0])<=c['K0']*e
                        products+=1
                        zero_D+=int(g['D']==0)
                    if z:
                        largest_ratios['q']=max(largest_ratios['q'],qerr/(c['Kq']*z))
                    if e:
                        largest_ratios['absolute']=max(largest_ratios['absolute'],Delta*Terr/(c['KT']*e))
                        largest_ratios['difference']=max(largest_ratios['difference'],herr/(c['KH']*e))
                    adversarial+=1
                truth_rows.append(dict(states=states,design=design_index,case=case,Delta=str(Delta),q_recovered=True,
                    contrasts_recovered=True,absolute_recovered=bool(Delta)))
    invalid=0
    for rows,ks,lo1,hi1,qmin1 in [([[F(1,2),F(1,2)]]*2,designs[0],lo,hi,F(1,10)),
        ([[F(0),F(0)]]*3,designs[0],lo,hi,F(1,10)),
        ([[F(1,2),F(1,2)]]*3,[F(1),F(1),F(2)],lo,hi,F(1,10)),
        ([[F(1,2),F(1,2)]]*3,designs[0],lo,hi,F(3,5))]:
        try:
            estimate(rows,ks,lo1,hi1,qmin1)
        except ValueError:
            invalid+=1
        else:
            raise AssertionError('Invalid input accepted')
    return dict(exact_truth_fixtures=len(truth_rows),flat_truths=flat,adversarial_tables=adversarial,
        projected_pair_checks=products,estimated_zero_denominators=zero_D,selected_pair_changes=switched,
        invalid_inputs_rejected=invalid,largest_fraction_of_conservative_bound={k:float(v) for k,v in largest_ratios.items()},
        fixtures=truth_rows)


def main():
    import argparse
    if sys.flags.optimize:
        raise RuntimeError('Do not disable scientific checks with -O')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    paths=['selection_risk.py','check_selection_risk.py']
    report=dict(checked_at=datetime.now(timezone.utc).isoformat(),status='FINITE_EXACT_DIAGNOSTICS_PASSED',
        symbolic_identities=symbolic(),checks=run(),sympy_version=sp.__version__,
        inputs=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in paths],
        scope='Finite checks of the selected manuscript implementation',limits='Finite synthetic checks do not prove minimax rates, unrestricted inequalities, formal correctness, novelty or importance.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report['checks'].items() if k!='fixtures'}))


if __name__=='__main__':
    main()

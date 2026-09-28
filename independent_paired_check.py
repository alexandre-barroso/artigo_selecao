"""Independent exact and bounded simulation audit; no producer imports/writes."""
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as R
from itertools import product
from math import factorial, comb, sqrt
from pathlib import Path
import hashlib
import json

import numpy as np

BASE = Path(__file__).resolve().parent
import argparse
import sys
if sys.flags.optimize:
    raise RuntimeError('Do not disable independent checks with -O')
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
OUT=parser.parse_args().output.resolve()
if OUT.exists():
    raise FileExistsError(OUT)
OUT.parent.mkdir(parents=True,exist_ok=True)
LOW, HIGH, QLOW = R(1,4), R(4), R(1,20)
SELECTED = {0, 5, 19, 35, 40, 60, 89, 95, 100, 119}


def project(value, low=LOW, high=HIGH):
    return min(high, max(low, value))


def inverse_labels(tables, k):
    """Independent direct implementation of v2 equations, exact rational input."""
    eta = QLOW*LOW/HIGH
    guarded = [[max(eta, x) for x in row] for row in tables]
    temporary = []
    for x in range(1, len(tables[0])):
        y = [row[x]/row[0] for row in guarded]
        slope_left = (y[1]-y[0])/(k[1]-k[0])
        slope_right = (y[2]-y[1])/(k[2]-k[1])
        den = k[2]*slope_right-k[0]*slope_left
        b = project((slope_left-slope_right)/den) if den else (LOW+HIGH)/2
        ratio = project((k[1]*y[0]-k[0]*y[1])/(k[1]-k[0])-
                        k[0]*k[1]*slope_left*b, QLOW, 1/QLOW)
        a = project((y[0]*(1+k[0]*b)/ratio-1)/k[0])
        temporary.append((den, ratio, a, y[0]))
    picked = sorted(range(len(temporary)), key=lambda j:(-abs(temporary[j][0]), j))[0]
    level0 = temporary[picked][2]
    rawq = [R(1)] + [entry[1] for entry in temporary]
    levels = [level0] + [project((entry[1]*(1+k[0]*level0)/entry[3]-1)/k[0])
                          for entry in temporary]
    return [x/sum(rawq) for x in rawq], levels


def inverse_retained(counts, totals, k, indices):
    i,j=indices
    a,b=k[i],k[j]
    floor=QLOW*(b-a)/((1+a*HIGH)*(1+b*HIGH))
    masses_a=[R(int(v),int(totals[i])) for v in counts[i]]
    masses_b=[R(int(v),int(totals[j])) for v in counts[j]]
    rawden=[b*right-a*left for left,right in zip(masses_a,masses_b)]
    levels=[project((left-right)/max(floor,d))
            for left,right,d in zip(masses_a,masses_b,rawden)]
    rawq=[project(mass*(1+a*t),QLOW,R(1)) for mass,t in zip(masses_a,levels)]
    diagnostics=dict(zero_cells=sum(v==0 for row in [counts[i],counts[j]] for v in row),
                     projected_denominators=sum(d<floor for d in rawden),
                     clipped_T_endpoints=sum(t==LOW or t==HIGH for t in levels))
    return ([x/sum(rawq) for x in rawq],levels),diagnostics


def exact_law_check():
    # Directly enumerate attempted strings, including the required final success.
    success=(R(1,5),R(3,10)); z=sum(success); failure=1-z
    tested=0
    for n in range(1,4):
        for failures in range(4):
            length=n+failures
            enumerated=defaultdict(R)
            for outcomes in product(range(3),repeat=length):
                if outcomes[-1]==2 or sum(x==2 for x in outcomes)!=failures:
                    continue
                weight=R(1)
                for x in outcomes: weight*=failure if x==2 else success[x]
                enumerated[(outcomes.count(0),outcomes.count(1))]+=weight
            for c0 in range(n+1):
                c1=n-c0
                multinomial=R(factorial(n),factorial(c0)*factorial(c1))
                factored=(comb(length-1,failures)*z**n*failure**failures
                          *multinomial*(success[0]/z)**c0*(success[1]/z)**c1)
                assert enumerated[(c0,c1)]==factored
                tested+=1
    return tested


def stats(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(np.mean(a)),
            'mc_standard_error':float(np.std(a,ddof=1)/sqrt(a.size))}


def loss(answer,q,t):
    qhat,that=answer
    # Square in exact arithmetic before conversion, unlike the producer.
    return {'q':float(max(abs(a-b) for a,b in zip(qhat,q))**2),
            'T':float(max(abs(a-b) for a,b in zip(that,t))**2),
            'mean_T':float((sum(that)/len(that)-sum(t)/len(t))**2)}


def main():
    logpath=BASE/'historical/retained_paired_simulation_v1.json'
    record=json.loads(logpath.read_text())
    assert record['seed']==202609274 and record['bit_generator']=='PCG64'
    assert record['numpy']==np.__version__
    input_hashes=[]
    source_map=json.loads((BASE/'historical/producer_source_map.json').read_text())
    for item in record['inputs']:
        mapped=source_map[item['path']]
        assert mapped['original_sha256']==item['sha256']
        digest=hashlib.sha256((BASE/mapped['portable_path']).read_bytes()).hexdigest()
        assert digest==mapped['portable_sha256']
        input_hashes.append(dict(original_sha256=item['sha256'],
            portable_path=mapped['portable_path'],portable_sha256=digest,
            scope='Original hash preserved; explicitly ported interface hash checked'))
    designs=[(R(1),R(2),R(5)),(R(1,50),R(1,2),R(40)),
             (R(1),R(1000001,1000000),R(20))]
    qs=[(R(1,2),)*2,(R(1,10),R(3,20),R(1,5),R(1,4),R(3,10))]
    cells=[]
    for q in qs:
        for design,k in enumerate(designs):
            for delta in [R(0),R(1,100),R(1,10),R(1)]:
                for N in [300,3000,30000,300000]:
                    cells.append((q,design,k,delta,R(3,2),N,'main'))
            for middle in [R(1,2),R(7,2)]:
                for N in [300,300000]:
                    cells.append((q,design,k,R(0),middle,N,'flat_level_control'))
    assert len(record['cells'])==len(cells)==120
    rng=np.random.Generator(np.random.PCG64(202609274))
    names=['accepted_only','retained_first_two','retained_endpoints']
    fields=['q','T','mean_T']
    point_worse={name:{field:[] for field in fields} for name in names[1:]}
    checks=[]; max_difference=0.0; endpoint_floor_checks=0
    all_effort_z=[]
    for ci,(q,design,k,delta,middle,N,role) in enumerate(cells):
        row=record['cells'][ci]; n=N//3;s=len(q)
        t=[middle+delta*(R(i,s-1)-R(1,2)) for i in range(s)]
        assert row['states']==s and row['design']==design and row['role']==role
        assert row['accepted_total']==N and row['accepted_per_regime']==n
        assert row['replicates']==200 and row['Delta']==str(delta)
        assert row['center']==str(middle) and row['q']==list(map(str,q))
        assert row['odds']==list(map(str,t)) and row['calibrations']==list(map(str,k))
        masses=[[qx/(1+kj*tx) for qx,tx in zip(q,t)] for kj in k]
        z=[sum(v) for v in masses]
        assert row['acceptance_probabilities']==list(map(str,z))
        expected=sum(R(n)/v for v in z)
        assert row['expected_total_attempts']==str(expected)
        floor=lambda i,j:QLOW*(k[j]-k[i])/((1+k[i]*HIGH)*(1+k[j]*HIGH))
        assert row['denominator_floors']['retained_first_two']==str(floor(0,1))
        assert row['denominator_floors']['retained_endpoints']==str(floor(0,2))
        assert floor(0,2)==max(floor(i,j) for i,j in [(0,1),(0,2),(1,2)])
        endpoint_floor_checks+=1
        for name in names:
            for field in fields:
                mean=row['risks'][name][field]['mean'];se=row['risks'][name][field]['mc_standard_error']
                assert np.isfinite(mean) and np.isfinite(se) and mean>=0 and se>=0
                assert mean<=float((HIGH-LOW)**2) if field!='q' else mean<=1
        for name in names[1:]:
            for field in fields:
                v=row['paired_loss_differences'][name][field]
                a=row['risks'][name][field];b=row['risks']['accepted_only'][field]
                assert abs(v['mean']-(a['mean']-b['mean']))<1e-12
                assert abs(a['mc_standard_error']-b['mc_standard_error'])<=v['mc_standard_error']+1e-12
                assert v['mc_standard_error']<=a['mc_standard_error']+b['mc_standard_error']+1e-12
                if v['mean']>0: point_worse[name][field].append(ci)
        effort=row['total_attempts']
        all_effort_z.append((effort['mean']-float(expected))/effort['mc_standard_error'])
        # Draw all cells to preserve RNG state; fit only the selected subset.
        counts=[rng.multinomial(n,[float(a/sum(v)) for a in v],size=200) for v in masses]
        attempts=[n+rng.negative_binomial(n,float(v),size=200) for v in z]
        regenerated_effort=stats(sum(attempts))
        for field in ['mean','mc_standard_error']:
            assert abs(regenerated_effort[field]-effort[field])<1e-8
        if ci not in SELECTED: continue
        scores={name:{field:[] for field in fields} for name in names}
        diagnostics={name:{'zero_cells':0,'projected_denominators':0,'clipped_T_endpoints':0}
                     for name in names[1:]}
        for repeat in range(200):
            observed_counts=[v[repeat] for v in counts]
            observed_totals=[v[repeat] for v in attempts]
            proportions=[[R(int(v),n) for v in entries] for entries in observed_counts]
            answers={'accepted_only':inverse_labels(proportions,k)}
            for name,indices in [('retained_first_two',(0,1)),('retained_endpoints',(0,2))]:
                answers[name],d=inverse_retained(observed_counts,observed_totals,k,indices)
                for key,value in d.items(): diagnostics[name][key]+=value
            for name,answer in answers.items():
                for field,value in loss(answer,q,t).items():scores[name][field].append(value)
        assert diagnostics==row['diagnostics']
        local_error=0.0
        for name in names:
            for field in fields:
                actual=stats(scores[name][field]);wanted=row['risks'][name][field]
                local_error=max(local_error,*[abs(actual[x]-wanted[x]) for x in actual])
                for key in actual: assert abs(actual[key]-wanted[key])<1e-11
                if name!='accepted_only':
                    paired=stats(np.array(scores[name][field])-np.array(scores['accepted_only'][field]))
                    wanted_pair=row['paired_loss_differences'][name][field]
                    local_error=max(local_error,*[abs(paired[x]-wanted_pair[x]) for x in paired])
                    for key in paired:assert abs(paired[key]-wanted_pair[key])<1e-11
        max_difference=max(max_difference,local_error)
        checks.append({'cell_index':ci,'states':s,'design':design,'Delta':str(delta),
                       'center':str(middle),'N':N,'replicates':200,
                       'max_absolute_summary_difference':local_error,'diagnostics_exact_match':True})
    result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
            'numpy':np.__version__,'seed':202609274,'exact_stopped_law_fixtures':exact_law_check(),
            'source_log_sha256':hashlib.sha256(logpath.read_bytes()).hexdigest(),
            'producer_inputs_verified':input_hashes,'planned_cells_verified':120,
            'all_attempt_summaries_regenerated':120,'all_risk_summary_consistency_checks':True,
            'endpoint_floor_checks':endpoint_floor_checks,
            'independent_estimator_replay_cells':checks,'replayed_replications':len(checks)*200,
            'max_absolute_replayed_summary_difference':max_difference,
            'positive_paired_difference_cell_counts':{n:{f:len(v) for f,v in values.items()} for n,values in point_worse.items()},
            'positive_paired_difference_cell_indices':point_worse,
            'maximum_absolute_total_attempts_mc_z':max(map(abs,all_effort_z)),
            'limits':['Ten cells independently re-fit; remaining risk means not claimed independently re-fit.',
                      'All120 grid entries, expected effort, effort draws and summary consistency checked.',
                      'Known-parameter synthetic experiment, floating random generation and exact rational estimator calculations.',
                      'Neither minimax theorem, priority nor importance follows from Monte Carlo comparisons.']}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['passed','exact_stopped_law_fixtures','planned_cells_verified',
                     'replayed_replications','max_absolute_replayed_summary_difference',
                     'positive_paired_difference_cell_counts','maximum_absolute_total_attempts_mc_z']},indent=2))


if __name__=='__main__':main()

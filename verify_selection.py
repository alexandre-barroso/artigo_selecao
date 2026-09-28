"""Portable selected-statistics reproduction entry point; failures are nonzero."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from release_inventory import verify_history

HERE=Path(__file__).resolve().parent
PINS={"numpy":"2.4.4","sympy":"1.14.0","mpmath":"1.3.0","matplotlib":"3.10.9","pillow":"12.2.0"}


def main():
    if sys.flags.optimize:
        raise RuntimeError("Do not disable scientific checks with -O")
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["audit","check","independent","simulation","figures","proofs","all"],default="check")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--mathlib",type=Path)
    p.add_argument("--clean-tree",action="store_true")
    a=p.parse_args()
    out=a.output.resolve()
    if out.is_relative_to(HERE) and not out.is_relative_to(HERE/"runs"):
        raise ValueError("Outputs inside this repository must be under runs/")
    if a.clean_tree and out.is_relative_to(HERE):
        raise ValueError("Clean audits write outside the repository")
    out.mkdir(parents=True,exist_ok=False)
    report={"started_at":datetime.now(timezone.utc).isoformat(),"mode":a.mode,"status":"RUNNING","steps":[],
            "limits":"Computational checks do not prove minimax rates, scientific novelty, empirical truth or hosted execution"}
    def save():
        (out/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    def call(script,arguments,step,timeout=1200):
        command=[sys.executable,"-B",str(HERE/script),*map(str,arguments)]
        env=os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
                   MPLCONFIGDIR=str(out/"runtime/matplotlib"),XDG_CACHE_HOME=str(out/"runtime/cache"))
        with (out/(step+".log")).open("w") as log:
            cp=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        report["steps"].append({"step":step,"exit_code":cp.returncode,"command":command})
        save()
        if cp.returncode:
            raise RuntimeError(step+" failed")
    save()
    try:
        report["inventory_files"]=verify_history(clean=a.clean_tree)
        if a.mode in {"proofs","all"} and not a.mathlib:
            raise ValueError("--mathlib is required for the requested proof scope")
        if a.mode!="audit":
            if sys.version_info[:2]!=(3,12):
                raise RuntimeError("Use Python 3.12; original environment was 3.12.3")
            report["Python"]=sys.version
            report["dependencies"]={k:importlib.metadata.version(k) for k in PINS}
            if report["dependencies"]!=PINS:
                raise RuntimeError("Pinned dependency mismatch")
        if a.mode in {"check","all"}:
            call("check_selection_risk.py",["--output",out/"accepted_controls.json"],"accepted_controls")
            call("check_selection_counts.py",["--output",out/"retained_controls.json"],"retained_controls")
            report["numerical_controls"]="Finite symbolic and exact rational fixtures passed"
        if a.mode in {"check","independent","all"}:
            call("independent_paired_check.py",["--output",out/"independent_checks.json"],"independent")
            independent=json.loads((out/"independent_checks.json").read_text())
            if not independent["passed"] or independent["replayed_replications"]!=2000 or independent["exact_stopped_law_fixtures"]!=36:
                raise RuntimeError("Independent implementation check failed")
            report["independent_check"]={k:independent[k] for k in ["replayed_replications","exact_stopped_law_fixtures","max_absolute_replayed_summary_difference","positive_paired_difference_cell_counts"]}
        if a.mode in {"simulation","all"}:
            call("simulate_retained_paired.py",["--output",out/"paired_simulation.json"],"simulation")
            old=json.loads((HERE/"historical/retained_paired_simulation_v1.json").read_text())
            new=json.loads((out/"paired_simulation.json").read_text())
            keys=["status","seed","numpy","bit_generator","domain","cells","limits"]
            matches={key:new[key]==old[key] for key in keys}
            report["simulation_replay"]={"compared_fields":matches,"cells":len(new["cells"]),
                                        "replicates_per_cell":200,"historical_timestamps_or_provenance_replaced":False}
            if not all(matches.values()):
                raise RuntimeError("Frozen simulation values were not reproduced exactly")
        if a.mode in {"figures","all"}:
            call("report_retained_paired.py",["--output",out/"artifacts"],"figures")
            checks=json.loads((HERE/"historical/artifact_checks.json").read_text())
            named=json.loads((HERE/"identity.json").read_text())["role"]=="named"
            names=["paired_all_cells.csv","paired_summary.tex","paired_numbers.tex","paired_risk.jpg"]
            if named:
                names.append("paired_risk.png")
            matches={name:hashlib.sha256((out/"artifacts"/name).read_bytes()).hexdigest()==checks["artifacts"][name]["sha256"] for name in names}
            if not all(matches.values()):
                raise RuntimeError("Reconstructed stable artifact mismatch")
            report["artifact_reconstruction"]={"stable_hash_matches":matches,
                "anonymous_png_limit":"Author chunk is neutralized; pixel identity checked separately",
                "PDF_limit":"Creation timestamp and anonymous author metadata may differ"}
        if a.mode in {"proofs","all"}:
            if not a.mathlib:
                raise ValueError("--mathlib is required for the requested proof scope")
            call("verify_selection_counts.py",["--mathlib",a.mathlib.resolve(),"--output",out/"proofs"],"proofs",timeout=480)
            proof=json.loads((out/"proofs/proofs.json").read_text())
            if proof["status"]!="PASSED_REQUESTED_SCOPE" or proof["declarations"]!=26:
                raise RuntimeError("Supporting proof inventory failed")
            report["proof_declarations"]=26
        report["status"]="PASSED_REQUESTED_SCOPE"
        code=0
    except Exception as exc:
        report.update(status="FAILED_OR_UNRESOLVED",error=type(exc).__name__+": "+str(exc))
        code=1
    report["completed_at"]=datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({k:report.get(k) for k in ["mode","status","error"]}),flush=True)
    return code


if __name__=="__main__":
    raise SystemExit(main())

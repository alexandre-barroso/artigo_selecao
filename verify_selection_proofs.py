"""Strict fresh builds of the two scoped supporting modules; no downloads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PIN = "5ed2965256430c3649e86755f9576b54eca72435"
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
TARGETS = {
    "SelectionStability": ["clip_mem", "clip_distance", "clippedRatio_mem", "interval_distance_bound",
        "projected_residual", "projected_product_bound", "firstDiff_identity",
        "denominator_identity", "numerator_identity", "slope_denominator_identity",
        "proposal_intercept_identity", "scalar_cancellation_bound", "affine_intercept_error",
        "absolute_perturbation", "selected_pair_bound", "calibration_fraction_bound"],
    "SelectionCounts": ["count_den_identity", "count_num_identity", "count_den_lower_bound",
        "lower_projection_distance", "projected_inverse_error", "count_den_error",
        "retained_level_error", "retained_level_model_error", "geometric_derivative_factor", "rate_estimate_error"],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(only_stability=False):
    if sys.flags.optimize:
        raise RuntimeError("Do not disable checks with -O")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathlib", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out, mathlib = args.output.resolve(), args.mathlib.resolve(strict=True)
    out.mkdir(parents=True, exist_ok=False)
    result = {"started_at":datetime.now(timezone.utc).isoformat(), "status":"RUNNING",
              "mathlib_commit":PIN, "modules":[]}
    def save():
        (out/"proofs.json").write_text(json.dumps(result,indent=2)+"\n")
    def probe(command):
        return subprocess.run(command,cwd=mathlib,capture_output=True,text=True,timeout=40,check=True).stdout.strip()
    save()
    try:
        version = probe(["lake","env","lean","--version"])
        if "version 4.34.0" not in version or probe(["git","rev-parse","HEAD"]) != PIN:
            raise ValueError("Pinned Lean/mathlib mismatch")
        prefix = probe(["lake","env","lean","--print-prefix"])
        imports = probe(["lake","env","printenv","LEAN_PATH"])
        result["Lean"] = version
        env = os.environ.copy()
        env["LEAN_PATH"] = str(out)+os.pathsep+imports
        names = ["SelectionStability"] if only_stability else list(TARGETS)
        for module in names:
            source = HERE/(module+".lean")
            raw = source.read_bytes()
            body = raw.decode()
            if re.search(r"^\s*(?:axiom|opaque|unsafe)\s|\b(?:sorry|admit|native_decide)\b",body,re.M):
                raise ValueError("Forbidden proof shortcut")
            if re.findall(r"^#print axioms (\w+)\s*$",body,re.M) != TARGETS[module]:
                raise ValueError("Formal target inventory changed")
            staged = out/source.name
            staged.write_bytes(raw)
            olean = out/(module+".olean")
            command = [str(Path(prefix)/"bin/lean"),"--trust=0","--root="+str(out),"-o",str(olean),str(staged)]
            cp = subprocess.run(command,cwd=mathlib,env=env,capture_output=True,text=True,timeout=180)
            log = out/(module+".log")
            text = cp.stdout+cp.stderr
            log.write_text(text)
            axioms = {n:[x.strip() for x in vals.split(",") if x.strip()]
                      for n,vals in re.findall(r"'([^']+)' depends on axioms:\s*\[([^]]*)\]",text)}
            for n in re.findall(r"'([^']+)' does not depend on any axioms",text):
                axioms[n]=[]
            passed = (cp.returncode==0 and olean.is_file() and source.read_bytes()==staged.read_bytes()==raw
                      and set(axioms)=={module+"."+x for x in TARGETS[module]}
                      and all(set(v)<=ALLOWED for v in axioms.values()))
            result["modules"].append({"module":module,"source_sha256":sha(source),"staged_source_sha256":sha(staged),
                "exit_code":cp.returncode,"passed":passed,"axioms":axioms,"log_sha256":sha(log),
                "olean_sha256":sha(olean) if olean.exists() else None,
                "command":command})
            save()
            if not passed:
                raise RuntimeError("Strict proof/axiom check failed: "+module)
        result.update(status="PASSED_REQUESTED_SCOPE",declarations=sum(len(r["axioms"]) for r in result["modules"]),
                      limits="Deterministic supporting obligations only; no sampling laws, integrated adaptive KL, minimax theorem or Python verification; no alternative kernel")
        code=0
    except Exception as exc:
        result.update(status="FAILED_OR_UNRESOLVED",error=type(exc).__name__+": "+str(exc))
        code=1
    result["completed_at"]=datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps({k:result.get(k) for k in ["status","declarations","error"]}),flush=True)
    return code


if __name__=="__main__":
    raise SystemExit(main())

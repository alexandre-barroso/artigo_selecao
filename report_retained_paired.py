"""Render the complete frozen paired diagnostic; no fitting or cell selection."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
import argparse
import sys
if sys.flags.optimize:
    raise RuntimeError('Do not disable scientific checks with -O')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
out = parser.parse_args().output.resolve()
out.mkdir(parents=True, exist_ok=False)
AUTHOR = json.loads((ROOT/'identity.json').read_text())['author']
os.environ.setdefault("MPLCONFIGDIR", str(out / "runtime/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(out / "runtime/cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE = ROOT / "historical/retained_paired_simulation_v1.json"
EXPECTED = "686712b307cefeaf6379ec3903b0bd5ff6ecee9fae6d7ca66eee9f9492ce280e"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
data = json.loads(SOURCE.read_text())
cells = data["cells"]
assert len(cells) == 120 and sum(c["role"] == "main" for c in cells) == 96

fields = ["cell", "role", "states", "design", "calibrations", "q", "T",
          "Delta", "accepted_total", "replicates", "method", "target",
          "risk", "risk_mcse", "paired_difference", "paired_mcse",
          "expected_attempts", "mean_attempts", "attempts_mcse"]
with (out / "paired_all_cells.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for i, cell in enumerate(cells):
        for method, risks in cell["risks"].items():
            for target, risk in risks.items():
                paired = cell["paired_loss_differences"].get(method, {}).get(target, {})
                writer.writerow({
                    "cell": i, "role": cell["role"], "states": cell["states"],
                    "design": cell["design"], "calibrations": ";".join(cell["calibrations"]),
                    "q": ";".join(cell["q"]), "T": ";".join(cell["odds"]),
                    "Delta": cell["Delta"], "accepted_total": cell["accepted_total"],
                    "replicates": cell["replicates"], "method": method, "target": target,
                    "risk": risk["mean"], "risk_mcse": risk["mc_standard_error"],
                    "paired_difference": paired.get("mean", ""),
                    "paired_mcse": paired.get("mc_standard_error", ""),
                    "expected_attempts": cell["expected_total_attempts"],
                    "mean_attempts": cell["total_attempts"]["mean"],
                    "attempts_mcse": cell["total_attempts"]["mc_standard_error"]})
methods = [("retained_first_two", "Primeiros dois regimes"),
           ("retained_endpoints", "Regimes extremos")]
rows = []
for key, label in methods:
    counts = [sum(c["paired_loss_differences"][key][t]["mean"] > 0 for c in cells)
              for t in ["q", "T", "mean_T"]]
    rows.append(label + " & " + " & ".join(f"{n}/120" for n in counts) + r" \\")
(out / "paired_summary.tex").write_text(
    "\\begin{table}[tb]\n\\centering\n"
    "\\caption{Cenários com diferença pontual de perda positiva: estimador "
    "com totais menos estimador reduzido. São contagens descritivas, "
    "não resultados de testes de hipóteses. Todos os 120 cenários entram "
    "nos denominadores.}\n\\label{tab:adversos}\n"
    "\\begin{tabular}{lrrr}\\toprule\n"
    "Estimador com totais & $q$ & $T$ & $\\tau$ \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")

def texnum(value, digits=5):
    return f"{value:.{digits}g}".replace(".", "{,}")

bad, good = cells[60], cells[95]
assert bad["states"] == good["states"] == 5
assert bad["design"] == 0 and bad["accepted_total"] == 300 and bad["Delta"] == "0"
assert good["design"] == 1 and good["accepted_total"] == 300000 and good["Delta"] == "1"
assert good["calibrations"] == ["1/50", "1/2", "40"]
numbers = {
    "BadReducedRisk": bad["risks"]["accepted_only"]["T"]["mean"],
    "BadRetainedRisk": bad["risks"]["retained_endpoints"]["T"]["mean"],
    "BadPairedDifference": bad["paired_loss_differences"]["retained_endpoints"]["T"]["mean"],
    "BadPairedMCSE": bad["paired_loss_differences"]["retained_endpoints"]["T"]["mc_standard_error"],
    "GoodReducedRisk": good["risks"]["accepted_only"]["T"]["mean"],
    "GoodFirstRisk": good["risks"]["retained_first_two"]["T"]["mean"],
    "GoodRetainedRisk": good["risks"]["retained_endpoints"]["T"]["mean"],
}
(out / "paired_numbers.tex").write_text("".join(
    "\\newcommand{\\" + name + "}{" + texnum(value) + "}\n" for name, value in numbers.items()))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(3, 2, figsize=(6.35, 6.65), layout="constrained")
Ns = [300, 3000, 30000, 300000]
deltas = [Fraction(0), Fraction(1, 100), Fraction(1, 10), Fraction(1)]
design_names = [r"$k=(1;2;5)$", r"$k=(0{,}02;0{,}5;40)$", r"$k=(1;1{,}000001;20)$"]
all_ratios = []
for design in range(3):
    for col, states in enumerate([2, 5]):
        ratios = np.empty((4, 4))
        for i, delta in enumerate(deltas):
            for j, n in enumerate(Ns):
                matches = [c for c in cells if c["role"] == "main"
                           and c["design"] == design and c["states"] == states
                           and c["accepted_total"] == n and Fraction(c["Delta"]) == delta]
                assert len(matches) == 1
                c = matches[0]
                ratios[i, j] = c["risks"]["retained_endpoints"]["T"]["mean"] / c["risks"]["accepted_only"]["T"]["mean"]
        assert np.all(np.isfinite(ratios)) and np.all(ratios > 0)
        all_ratios.extend(ratios.ravel().tolist())
        ax = axes[design, col]
        shown = np.log10(ratios)
        im = ax.imshow(shown, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        for i in range(4):
            for j in range(4):
                color = "white" if abs(shown[i, j]) > 1.5 else "black"
                ax.text(j, i, f"{ratios[i,j]:.2g}".replace(".", ","),
                        ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(range(4), ["300", "3 mil", "30 mil", "300 mil"])
        ax.set_yticks(range(4), ["0", "0,01", "0,1", "1"])
        ax.set_title(f"$s={states}$; " + design_names[design], fontsize=9)
        ax.set_ylabel(r"Amplitude $\Delta$")
        ax.set_xlabel("Aceitações N")
fig.colorbar(im, ax=axes, location="bottom", shrink=.82, pad=.035,
             label="log₁₀ da razão de riscos (totais / reduzido)", extend="both")
metadata = {"Title": "Risco pareado de níveis de seleção",
            "Author": AUTHOR, "Subject": "Diagnóstico sintético"}
fig.savefig(out / "paired_risk.pdf", metadata=metadata)
fig.savefig(out / "paired_risk.png", dpi=400, metadata={"Author": AUTHOR})
fig.savefig(out / "paired_risk.jpg", dpi=400, pil_kwargs={"quality": 95})
plt.close(fig)
report = {"input_sha256": EXPECTED, "all_cells": len(cells),
          "plotted_main_cells": len(all_ratios), "plotted_ratio_min": min(all_ratios),
          "plotted_ratio_max": max(all_ratios), "numbers": numbers,
          "figure_scope": "Full-T endpoint/accepted risk ratios; 96 main cells. 24 additional flat controls retained in CSV and table.",
          "color_scope": "Fixed log10 limits [-3,3], raw ratio labels; colorbar marks clipping at either end.",
          "artifacts": [{"path": str(p.relative_to(out)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]}
(out / "artifact_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"all_cells": len(cells), "plotted_cells": len(all_ratios), "status": "GENERATED"}))

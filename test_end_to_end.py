#!/usr/bin/env python3
"""End-to-end tests for the MixLab pipeline (P0-FR01 … P0-FR05).

Run:  python3 tests/test_end_to_end.py
No pytest required; exits non-zero on the first failed assertion.

Covers:
  1. valid config  -> exit 0, candidates produced, constraints respected
  2. missing target_column -> exit 2, BLOCKED_INPUT, no candidate file
  3. bounds conflict (lower >= upper) -> exit 2, BLOCKED_INPUT
  4. determinism: two identical runs give identical predictions
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
RUNNER = SKILL / "scripts" / "run_mixlab.py"
SAMPLE = SKILL / "assets" / "sample_input_SIMULATED.csv"
CONFIG = SKILL / "config" / "example_config.json"

failures = []


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUNNER), *map(str, args)],
                          capture_output=True, text=True)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def main() -> int:
    work = Path.cwd() / ".mixlab_test_runs"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        # 1. valid run
        out1 = work / "valid"
        r = run("--input", SAMPLE, "--config", CONFIG, "--outdir", out1)
        check("valid run exit 0", r.returncode == 0, r.stdout.strip()[:200] + r.stderr.strip()[:200])
        for name in ["run_config.json", "data_audit.json", "model_metrics.json",
                     "predictions.csv", "candidate_experiments.csv",
                     "research_report.md", "run_manifest.json"]:
            check(f"valid run produces {name}", (out1 / name).exists())
        for fig in ["data_overview.png", "target_distribution.png", "pred_vs_actual.png",
                    "candidate_comparison.png", "candidate_ranking.png"]:
            check(f"valid run produces figures/{fig}", (out1 / "figures" / fig).exists())

        manifest = json.loads((out1 / "run_manifest.json").read_text())
        check("manifest status OK", manifest["status"] == "OK")
        check("manifest records exit code 0", manifest["exit_code"] == 0)
        check("manifest records input sha256", len(manifest["input"]["sha256"]) == 64)

        cfg = json.loads(CONFIG.read_text())
        top_k = cfg["recommendation"]["top_k"]
        import pandas as pd
        cand = pd.read_csv(out1 / "candidate_experiments.csv")
        check(f"candidate count == top_k ({top_k})", len(cand) == top_k, f"got {len(cand)}")
        check("candidate ranks are 1..k", sorted(cand["rank"].tolist()) == list(range(1, top_k + 1)))
        within = True
        for var, spec in cfg["variables"].items():
            if spec["role"] != "controllable":
                continue
            lo, hi = spec["bounds"]
            within &= bool(((cand[var] >= lo) & (cand[var] <= hi)).all())
        check("all candidates respect user bounds", within)
        check("candidates are labelled UNVERIFIED_RECOMMENDATION",
              bool((cand["data_label"] == "UNVERIFIED_RECOMMENDATION").all()))

        # 2. missing target_column
        bad = json.loads(CONFIG.read_text())
        bad.pop("target_column")
        bad_path = work / "bad_no_target.json"
        bad_path.write_text(json.dumps(bad))
        out2 = work / "no_target"
        r2 = run("--input", SAMPLE, "--config", bad_path, "--outdir", out2)
        check("missing target_column -> exit 2", r2.returncode == 2, f"got {r2.returncode}")
        check("missing target_column -> BLOCKED_INPUT",
              json.loads((out2 / "run_manifest.json").read_text())["status"] == "BLOCKED_INPUT")
        check("missing target_column -> no candidate file",
              not (out2 / "candidate_experiments.csv").exists())

        # 3. bounds conflict
        bad2 = json.loads(CONFIG.read_text())
        bad2["variables"]["temperature_C"]["bounds"] = [140.0, 90.0]
        bad2_path = work / "bad_bounds.json"
        bad2_path.write_text(json.dumps(bad2))
        out3 = work / "bad_bounds"
        r3 = run("--input", SAMPLE, "--config", bad2_path, "--outdir", out3)
        check("bounds conflict -> exit 2", r3.returncode == 2, f"got {r3.returncode}")
        check("bounds conflict -> message mentions bounds",
              "bounds" in json.loads((out3 / "run_manifest.json").read_text())["blocking_reason"])

        # 4. determinism
        out4 = work / "repeat"
        run("--input", SAMPLE, "--config", CONFIG, "--outdir", out4)
        a = (out1 / "predictions.csv").read_text()
        b = (out4 / "predictions.csv").read_text()
        check("repeat run reproduces predictions.csv", a == b)

        # 5. replay validation on the positive control (known linear structure)
        replay_out = work / "replay"
        rr = subprocess.run(
            [sys.executable, str(SKILL / "scripts" / "replay_validate.py"),
             "--input", str(SKILL / "assets" / "control_sim_240.csv"),
             "--config", str(SKILL / "config" / "control_sim_replay.json"),
             "--outdir", str(replay_out), "--repeats", "3", "--top-k", "3"],
            capture_output=True, text=True)
        check("replay runs on control data", rr.returncode == 0, rr.stderr.strip()[:200])
        rv = json.loads((replay_out / "replay_validation.json").read_text())
        check("replay reports a verdict", rv["verdict"] in
              {"EVIDENCE_OF_ADVANTAGE", "WEAK_OR_INCONCLUSIVE", "NO_EVIDENCE_OVER_RANDOM"})
        check("replay detects advantage on linear control",
              rv["verdict"] == "EVIDENCE_OF_ADVANTAGE", rv["verdict"])
        check("replay produces comparison figure",
              (replay_out / "figures" / "replay_comparison.png").exists())
        check("replay outputs report + manifest",
              (replay_out / "replay_report.md").exists() and
              (replay_out / "run_manifest.json").exists())

        print("\n" + ("ALL TESTS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
        return 0 if not failures else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

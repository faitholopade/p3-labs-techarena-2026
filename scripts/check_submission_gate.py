#!/usr/bin/env python3
"""Read-only screen for using a scarce competition submission slot."""

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def number(value, label, auc=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (auc and not 0 <= result <= 1):
        raise ValueError(f"{label} is not a valid finite value")
    return result


def evaluate(report, bootstrap, history, policy, candidate, remaining):
    if not 0 <= remaining <= policy["daily_submission_limit"]:
        raise ValueError("remaining slots must be between 0 and the daily limit")
    scored = [row for row in history if row.get("official_auc") not in (None, "")]
    if not scored:
        raise ValueError("no scored incumbent in the official history")
    best = max(scored, key=lambda row: number(row["official_auc"], "official_auc", auc=True))
    incumbent = best["version"]
    threshold_key = (
        "scarce_slots_minimum_local_gain"
        if remaining <= policy["scarce_slots_at_most"]
        else "normal_minimum_local_gain"
    )
    threshold = number(policy[threshold_key], threshold_key)
    if threshold <= 0:
        raise ValueError("minimum gain must be positive")
    slices = policy["required_positive_slices"]
    gains = {}
    row_counts = {}
    for name in ["combined", *slices]:
        block = report[name]
        count = block["rows"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name}.rows must be a positive integer")
        row_counts[name] = count
        baseline_auc = number(block[incumbent]["auc"], f"{name}.{incumbent}.auc", auc=True)
        candidate_auc = number(block[candidate]["auc"], f"{name}.{candidate}.auc", auc=True)
        gains[name] = candidate_auc - baseline_auc
    if row_counts["combined"] != sum(row_counts[name] for name in slices):
        raise ValueError("combined row count must equal the sum of the disjoint date slices")
    bootstrap_gain = number(bootstrap["auc_improvement"], "bootstrap.auc_improvement")
    if not math.isclose(bootstrap_gain, gains["combined"], abs_tol=1e-10, rel_tol=0):
        raise ValueError("bootstrap gain does not match this candidate versus the official incumbent")
    interval = bootstrap["interval_95_percent"]
    if len(interval) != 2:
        raise ValueError("bootstrap interval must contain two bounds")
    lower, upper = [number(value, "bootstrap interval") for value in interval]
    if lower > upper:
        raise ValueError("bootstrap interval bounds are reversed")
    reasons = []
    if remaining == 0:
        reasons.append("No submission slots remain in the supplied portal count.")
    if gains["combined"] < threshold and not math.isclose(
        gains["combined"], threshold, abs_tol=1e-12, rel_tol=0
    ):
        reasons.append(f"Combined local gain is below {threshold:.6f}.")
    for name in slices:
        if gains[name] <= 0:
            reasons.append(f"The {name} date slice does not improve.")
    if policy["require_positive_paired_user_bootstrap_lower_95_bound"] and lower <= 0:
        reasons.append("The paired bootstrap lower bound is not positive.")
    return {
        "decision": "hold" if reasons else "eligible_for_submission_review",
        "candidate": candidate,
        "incumbent": incumbent,
        "incumbent_official_auc": number(best["official_auc"], "official_auc", auc=True),
        "remaining_slots_supplied": remaining,
        "minimum_local_gain": threshold,
        "local_gains": gains,
        "paired_bootstrap_95_interval": [lower, upper],
        "reasons": reasons,
        "caveat": "This is a provisional screen, not a forecast or an upload. The caller must supply matched predictions and a paired user bootstrap against the same incumbent; this reader cannot establish their provenance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--bootstrap", required=True, type=Path)
    parser.add_argument("--remaining", required=True, type=int, help="Current count from the portal")
    parser.add_argument("--history", type=Path, default=ROOT / "results/score_history.csv")
    parser.add_argument("--policy", type=Path, default=ROOT / "config/submission_policy.json")
    args = parser.parse_args()
    try:
        with args.history.open() as stream:
            history = list(csv.DictReader(stream))
        result = evaluate(
            json.loads(args.report.read_text()),
            json.loads(args.bootstrap.read_text()),
            history,
            json.loads(args.policy.read_text()),
            args.candidate,
            args.remaining,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"decision": "needs_valid_evidence", "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "eligible_for_submission_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())

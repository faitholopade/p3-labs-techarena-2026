"""Fresh chronological factor backtest; no saved calibration heads are reused.

This intentionally does not claim to reproduce V20. It creates a common factor
baseline at a new cutoff so weekday evidence can be collected without using
Saturday/Sunday factors that have already seen the weekday outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from scipy.special import softmax
import torch

from common import binary_labels, load_data, metrics, split_data, subset
from posterior_inference import features as posterior_features
from projected import state as projected_state
import v24_item_time


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_cut(value):
    if str(value).isdigit():
        return int(value)
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fit_command(arguments, folder, expected):
    checkpoint = folder / expected
    record = folder / "training.json"
    done = folder / "completed.json"
    if done.exists():
        saved = json.loads(done.read_text())
        if not checkpoint.is_file() or sha256(checkpoint) != saved["checkpoint_sha256"]:
            raise ValueError(f"Completed checkpoint failed its digest: {checkpoint}")
        if not record.is_file() or sha256(record) != saved["training_sha256"]:
            raise ValueError(f"Completed training record failed its digest: {record}")
        return checkpoint
    if checkpoint.exists() or record.exists():
        raise ValueError(f"Unverified interrupted output in {folder}; use a new output directory.")
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable] + [str(x) for x in arguments]
    print(json.dumps({"stage": "fit", "command": command, "log": str(folder / "fit.log")}), flush=True)
    started = time.monotonic()
    with (folder / "fit.log").open("w") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not checkpoint.is_file() or not record.is_file():
        raise RuntimeError("Training returned without both checkpoint and training record.")
    write_json(done, {"command": command, "seconds": time.monotonic() - started,
                     "checkpoint_sha256": sha256(checkpoint), "training_sha256": sha256(record)})
    return checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="V11 runtime containing data/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cut", type=parse_cut, required=True, help="UTC date YYYY-MM-DD or epoch seconds")
    p.add_argument("--variant", choices=["core", "joint"], default="joint")
    p.add_argument("--joint-epochs", type=int, default=24)
    p.add_argument("--horizon-days", type=int, default=1,
                   help="Retain first query per user within this horizon; zero retains all later dates")
    p.add_argument("--posterior-samples", type=int, default=256,
                   help="Shared Sobol draws plus their opposites")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--prepare-only", action="store_true", help="Prepare cutoff-only bases, without training")
    p.add_argument("--item-time", action="store_true", help="Fit the fixed per-item cyclic hour candidate")
    a = p.parse_args()
    if a.horizon_days < 0 or a.posterior_samples < 1 or a.joint_epochs < 1 or a.threads < 1:
        raise ValueError("Invalid schedule, horizon, samples, or thread count.")
    a.root = a.root.resolve(); a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    source_paths = ["scripts/backtest_v24_factor.py", "src/common.py", "src/projected.py",
                    "src/train_factor_core.py", "src/refine_factor_joint.py", "src/variational.py",
                    "src/residual_model.py", "src/posterior_inference.py", "src/negative.py",
                    "src/cluster_unit.py", "src/uncertainty.py", "src/topic.py", "src/latent.py",
                    "src/v24_item_time.py"]
    plan = {"cut": a.cut, "root": str(a.root), "variant": a.variant,
            "joint_epochs": a.joint_epochs, "horizon_days": a.horizon_days,
            "posterior_samples": a.posterior_samples, "threads": a.threads, "item_time": a.item_time,
            "data_sha256": sha256(a.root / "data/train.npz"),
            "audit_sha256": sha256(a.root / "data/audit.json"),
            "source_sha256": {name: sha256(REPO / name) for name in source_paths},
            "core_recipe": {"preference_epochs": 16, "residual_epochs": 17, "seed": 3951,
                            "preference_kl": 0.25, "warm_weight": 1.0, "threads": 3},
            "joint_recipe": {"epochs": a.joint_epochs, "lr": 0.001, "seed": 8819,
                             "warm_weight": 1.0, "threads": a.threads},
            "selection": "Fixed schedules; no checkpoint, seed, or blend selection.",
            "scope": "Standalone factor backtest, not a reproduction of V20 or its validation score."}
    plan_path = a.out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing output belongs to a different plan, data, or source version.")
    write_json(plan_path, plan)
    data, evaluation, audit = load_data(a.root / "data")
    del evaluation
    train_indices, query_indices = split_data(data, a.cut)
    if a.horizon_days:
        query_indices = query_indices[data["timestamp"][query_indices] < a.cut + 86400 * a.horizon_days]
    if not len(train_indices) or not len(query_indices):
        raise ValueError("The cutoff/horizon must leave nonempty training and query sets.")
    fit = subset(data, train_indices)
    query = subset(data, query_indices)
    labels = binary_labels(query)
    query["target"] = np.full(len(query_indices), -1, dtype=np.int64)
    assert fit["timestamp"].max() < a.cut <= query["timestamp"].min()
    assert len(np.unique(query["user"])) == len(query_indices)
    hm = query["hist"] > 0
    if not np.all(query["hist_ts"][hm] < np.broadcast_to(query["timestamp"][:, None], hm.shape)[hm]):
        raise ValueError("Query history contains current/future events.")
    if not np.all(query["hist_ts"][hm] < a.cut):
        raise ValueError("A first-post-cutoff query has unexpected post-cutoff history.")
    print(json.dumps({"stage": "prepare", "training_events": len(train_indices),
                      "query_events": len(query_indices), "cut": a.cut}), flush=True)
    bases = a.out / "bases"
    basis_done = bases / "completed.json"
    init_path = bases / "p0_u0.5_i0.5_r5.npy"
    exposure_path = bases / "exposure.npz"
    if basis_done.exists():
        hashes = json.loads(basis_done.read_text())
        for path in [init_path, exposure_path]:
            if not path.is_file() or sha256(path) != hashes[path.name]:
                raise ValueError(f"Fresh-cutoff basis failed its digest: {path}")
    else:
        if bases.exists() and any(bases.iterdir()):
            raise ValueError("Unverified partial basis cache exists; use a new output directory.")
        projected_state(fit, audit["n_items"], audit["n_users"], bases, 0, .5, .5, 5)
        write_json(basis_done, {path.name: sha256(path) for path in [init_path, exposure_path]})
    cohort = {"training_events": len(train_indices), "query_events": len(query_indices),
              "max_training_timestamp": int(fit["timestamp"].max()),
              "min_query_timestamp": int(query["timestamp"].min()),
              "max_query_timestamp": int(query["timestamp"].max()),
              "weekday_events": int(np.sum((query["timestamp"] // 86400 + 3) % 7 < 5)),
              "query_labels_masked": True, "history_before_cutoff": True,
              "calibration_heads_used": False, "future_factors_used": False}
    write_json(a.out / "cohort.json", cohort)
    del data
    if not a.item_time:
        del fit
    gc.collect()
    if a.prepare_only:
        return
    core = fit_command([REPO / "src/train_factor_core.py", "--root", a.root,
                        "--init", init_path, "--exposure", exposure_path,
                        "--pool-cache", a.out / "pools", "--cut", a.cut,
                        "--out", a.out / "core", "--seed", 3951,
                        "--kl", .25, "--warm-weight", 1.], a.out / "core", "residual.pt")
    variants = {"core": core}
    if a.variant == "joint":
        variants["joint"] = fit_command(
            [REPO / "src/refine_factor_joint.py", "--root", a.root, "--init", core,
             "--exposure", exposure_path, "--pool-cache", a.out / "pools", "--cut", a.cut,
             "--out", a.out / "joint", "--epochs", a.joint_epochs, "--lr", .001,
             "--seed", 8819, "--warm-weight", 1., "--threads", a.threads],
            a.out / "joint", "residual.pt")
    saved_predictions = {}
    reports = {}
    for name, checkpoint in variants.items():
        print(json.dumps({"stage": "score", "variant": name}), flush=True)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(ck["cut"]) != a.cut:
            raise ValueError("Checkpoint cutoff disagrees with the chronological split.")
        probability = v24_item_time.posterior_probability(ck["state"], query, samples=a.posterior_samples)
        if not np.isfinite(probability).all() or np.max(probability.sum(1)) > 1 + 1e-7:
            raise ValueError("Invalid probability output.")
        saved_predictions[name] = probability.astype(np.float64)
        reports[name] = {**metrics(labels, probability), "checkpoint_sha256": sha256(checkpoint)}
        reports[name]["first_24_hours"] = metrics(labels[query["timestamp"] < a.cut + 86400],
                                                  probability[query["timestamp"] < a.cut + 86400])
        reports[name]["by_query_day"] = {
            str(int(day)): metrics(labels[query["timestamp"] // 86400 == day],
                                   probability[query["timestamp"] // 86400 == day])
            for day in np.unique(query["timestamp"] // 86400)}
        if a.item_time and name == a.variant:
            candidate_dir = a.out / "item_time"
            if (candidate_dir / "model.pt").exists():
                candidate = torch.load(candidate_dir / "model.pt", map_location="cpu", weights_only=False)
                if candidate['spec']['factor_sha256'] != sha256(checkpoint) or candidate['spec']['cut'] != a.cut:
                    raise ValueError("Item-time checkpoint has different chronological factor provenance.")
            else:
                fitting_probability = v24_item_time.posterior_probability(ck["state"], fit, samples=a.posterior_samples)
                candidate = v24_item_time.train(fit, fitting_probability, audit['n_items'], candidate_dir,
                                                sha256(checkpoint), a.cut, a.threads)
                del fitting_probability
            changed, checks = v24_item_time.predict(candidate, query, probability)
            saved_predictions['item_time'] = changed
            reports['item_time'] = {**metrics(labels, changed), 'checks': checks,
                                   'first_24_hours': metrics(labels[query['timestamp'] < a.cut + 86400],
                                                             changed[query['timestamp'] < a.cut + 86400]),
                                   'by_query_day': {str(int(day)): metrics(labels[query['timestamp'] // 86400 == day],
                                                                          changed[query['timestamp'] // 86400 == day])
                                                    for day in np.unique(query['timestamp'] // 86400)}}
            reports['item_time_minus_parent_auc'] = reports['item_time']['auc'] - reports[name]['auc']
            reports['item_time_minus_parent_first_24h_auc'] = (reports['item_time']['first_24_hours']['auc']
                                                              - reports[name]['first_24_hours']['auc'])
        del ck
        gc.collect()
    np.savez_compressed(a.out / "predictions.npz", indices=query_indices, labels=labels,
                        user=query["user"], timestamp=query["timestamp"], context=query["context"],
                        **saved_predictions)
    if "joint" in reports:
        reports["joint_minus_core_auc"] = reports["joint"]["auc"] - reports["core"]["auc"]
    write_json(a.out / "report.json", {"plan": plan, "cohort": cohort, "metrics": reports,
               "prediction_sha256": sha256(a.out / "predictions.npz"),
               "limitation": "No exact V20 prediction is reconstructed for this cutoff. These are standalone factor scores."})
    print(json.dumps({"stage": "complete", "metrics": reports}), flush=True)


if __name__ == "__main__":
    main()

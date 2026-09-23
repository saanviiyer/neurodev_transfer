"""Encoder-level analysis of functional transfer across experience curricula."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import links as L


ORDER_CURRICULA = ("slow_first", "fast_first", "shuffled")
CONTRASTS = (("slow_first", "fast_first"),
             ("slow_first", "shuffled"),
             ("fast_first", "shuffled"))


def load_policy_rows(indir: Path):
    return [json.loads(p.read_text()) for p in sorted(indir.glob("*.json"))]


def aggregate(rows, required_policy_seeds):
    grouped = {}
    for row in rows:
        key = (row["kind"], row["curriculum"], int(row["encoder_seed"]))
        grouped.setdefault(key, []).append(row)
    encoders = []
    required = set(required_policy_seeds)
    for (kind, curriculum, seed), group in sorted(grouped.items()):
        present = {int(r["policy_seed"]) for r in group}
        if present != required:
            raise ValueError(f"{kind}/{curriculum}/s{seed}: policy seeds "
                             f"{sorted(present)}, expected {sorted(required)}")
        encoders.append({
            "kind": kind, "curriculum": curriculum, "encoder_seed": seed,
            "n_policy_seeds": len(group),
            "final_success": float(np.mean([r["final"]["success"] for r in group])),
            "final_spl": float(np.mean([r["final"]["spl"] for r in group])),
            "final_progress": float(np.mean([r["final"]["progress"] for r in group])),
            "success_auc": float(np.mean([r["success_auc"] for r in group])),
            "policy_seed_success_sd": float(np.std(
                [r["final"]["success"] for r in group], ddof=0)),
            "source_sha256": sorted({r["source_sha256"] for r in group}),
        })
    return encoders


def paired(encoders, a, b, metric, bootstrap=20_000, seed=811):
    index = {(r["kind"], r["encoder_seed"], r["curriculum"]): r for r in encoders}
    keys = sorted({(k, s) for k, s, c in index if c == a}
                  & {(k, s) for k, s, c in index if c == b})
    pairs = [{"kind": k, "encoder_seed": s,
              "a": index[k, s, a][metric], "b": index[k, s, b][metric]}
             for k, s in keys]
    differences = np.asarray([p["a"] - p["b"] for p in pairs], dtype=float)
    if not len(differences):
        return {"n_pairs": 0, "status": "INCOMPLETE", "pairs": []}
    rng = np.random.default_rng(seed)
    draws = differences[rng.integers(
        0, len(differences), size=(bootstrap, len(differences)))].mean(1)
    return {
        "n_pairs": len(pairs),
        "status": "PASSABLE" if len(pairs) >= 8 else "MORE_SEEDS",
        "mean_difference": float(differences.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [.025, .975])],
        "exact_p": L.stat_tests.exact_sign_flip_p(differences),
        "all_same_sign": bool(np.all(differences > 0) or np.all(differences < 0)),
        "pairs": pairs,
    }


def join_development(encoders, development_path: Path):
    development = json.loads(development_path.read_text())
    dev = {(r["kind"], r["curriculum"], int(r["seed"])): r
           for r in development["per_run"]}
    out = []
    for row in encoders:
        key = (row["kind"], row["curriculum"], row["encoder_seed"])
        if key not in dev:
            raise ValueError(f"missing developmental analysis row: {key}")
        joined = dict(row)
        joined["development"] = dev[key]["emergence"]
        out.append(joined)
    return out


def descriptive_relationships(rows):
    """Descriptive only: clustered curriculum rows do not justify naive p-values."""
    out = {}
    y = np.asarray([r["success_auc"] for r in rows])
    for cap in ("heading_decode", "heading", "place", "long_gap"):
        for field in ("step", "asymptote"):
            x = np.asarray([np.nan if r["development"][cap][field] is None
                            else r["development"][cap][field] for r in rows], float)
            keep = np.isfinite(x) & np.isfinite(y)
            if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
                corr = None
            else:
                # Rank correlation without adding a dependency on scipy.
                xr = np.argsort(np.argsort(x[keep])).astype(float)
                yr = np.argsort(np.argsort(y[keep])).astype(float)
                corr = float(np.corrcoef(xr, yr)[0, 1])
            out[f"{cap}_{field}_vs_success_auc"] = {
                "spearman_approx": corr, "n_rows": int(keep.sum()),
                "inferential": False,
            }
    return out


def validate_matrix(encoders, required_curricula=ORDER_CURRICULA,
                    required_kinds=("gru", "rnn"), required_encoder_seeds=(0, 1, 2, 3)):
    expected = {(k, c, s) for k in required_kinds for c in required_curricula
                for s in required_encoder_seeds}
    found = {(r["kind"], r["curriculum"], r["encoder_seed"]) for r in encoders}
    return sorted(expected - found)


def main(indir, output, development, required_policy_seeds, allow_incomplete):
    encoders = aggregate(load_policy_rows(indir), required_policy_seeds)
    joined = join_development(encoders, development)
    missing = validate_matrix(encoders)
    if missing and not allow_incomplete:
        raise SystemExit(f"incomplete confirmatory matrix: {missing}")
    comparisons = {}
    for a, b in CONTRASTS:
        for metric in ("final_success", "success_auc", "final_spl",
                       "final_progress"):
            comparisons[f"{a}_vs_{b}_{metric}"] = paired(encoders, a, b, metric)
    result = {
        "status": "INCOMPLETE" if missing else
                  ("MORE_SEEDS" if any(v["n_pairs"] < 8 for v in comparisons.values())
                   else "COMPLETE"),
        "missing_confirmatory_cells": missing,
        "encoder_rows": joined,
        "comparisons": comparisons,
        "descriptive_relationships": descriptive_relationships(joined),
        "excluded_primary_contrast": {
            "pair": ["deprived_early", "deprived_late"],
            "reason": "unequal post-deprivation recovery time in source design"
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(result["status"], output)
    for name, row in comparisons.items():
        if row["n_pairs"]:
            print(name, round(row["mean_difference"], 4), row["ci95"],
                  f"p={row['exact_p']:.5f}", f"n={row['n_pairs']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("runs/confirmatory"))
    ap.add_argument("--output", type=Path, default=Path("runs/transfer_analysis.json"))
    ap.add_argument("--development", type=Path, default=L.DEVELOPMENT_JSON)
    ap.add_argument("--required-policy-seeds", default="0,1,2")
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    main(a.indir, a.output, a.development,
         [int(x) for x in a.required_policy_seeds.split(",")], a.allow_incomplete)


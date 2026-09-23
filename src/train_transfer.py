"""Train reward-only policies on frozen developmental world-model states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import links as L


def parse_checkpoint_tag(tag: str):
    """Parse tags such as `gru_deprived_early_s2` without losing underscores."""
    parts = tag.split("_")
    if len(parts) < 3 or not parts[-1].startswith("s"):
        raise ValueError(f"invalid developmental checkpoint tag: {tag}")
    return parts[0], "_".join(parts[1:-1]), int(parts[-1][1:])


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_checkpoints(indir: Path, kinds, curricula, encoder_seeds):
    rows = []
    for path in sorted(indir.glob("*.pt")):
        kind, curriculum, seed = parse_checkpoint_tag(path.stem)
        if kind in kinds and curriculum in curricula and seed in encoder_seeds:
            rows.append((path, kind, curriculum, seed))
    expected = {(k, c, s) for k in kinds for c in curricula for s in encoder_seeds}
    found = {(k, c, s) for _, k, c, s in rows}
    missing = sorted(expected - found)
    if missing:
        raise SystemExit(f"missing source checkpoints: {missing}")
    return rows


def main(indir: Path, outdir: Path, kinds, curricula, encoder_seeds,
         policy_seeds, updates: int, batch: int, horizon: int):
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoints = selected_checkpoints(indir, kinds, curricula, encoder_seeds)
    for path, kind, curriculum, encoder_seed in checkpoints:
        model = L.models.build(kind)
        model.load_state_dict(torch.load(path, map_location="cpu")); model.eval()
        source_hash = sha256(path)
        for policy_seed in policy_seeds:
            tag = f"{kind}_{curriculum}_s{encoder_seed}_p{policy_seed}"
            result_path, weights_path = outdir / f"{tag}.json", outdir / f"{tag}.pt"
            if result_path.exists() and weights_path.exists():
                print("skip", tag, flush=True)
                continue
            print(f"\n{tag}", flush=True)
            policy, result = L.rl_navigation.fit_policy(
                model, policy_seed, "rectangle", updates, batch, horizon)
            payload = {
                "tag": tag, "kind": kind, "curriculum": curriculum,
                "encoder_seed": encoder_seed, "policy_seed": policy_seed,
                "source_checkpoint": str(path.resolve()),
                "source_sha256": source_hash, "encoder_frozen": True,
                "pose_policy_input": False, "domain": "rectangle",
                "updates": updates, "batch": batch, "horizon": horizon,
                **result,
            }
            result_path.write_text(json.dumps(payload, indent=2))
            torch.save(policy.state_dict(), weights_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=L.DEV_RUNS)
    ap.add_argument("--outdir", type=Path, default=Path("runs/confirmatory"))
    ap.add_argument("--kinds", default="gru,rnn")
    ap.add_argument("--curricula", default="slow_first,fast_first,shuffled")
    ap.add_argument("--encoder-seeds", default="0,1,2,3")
    ap.add_argument("--policy-seeds", default="0,1,2")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=45)
    a = ap.parse_args()
    main(a.indir, a.outdir, set(a.kinds.split(",")),
         set(a.curricula.split(",")), {int(x) for x in a.encoder_seeds.split(",")},
         [int(x) for x in a.policy_seeds.split(",")], a.updates, a.batch, a.horizon)


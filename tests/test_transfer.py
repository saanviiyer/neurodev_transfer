import json

import pytest

from analyze_transfer import aggregate, paired, validate_matrix
from train_transfer import parse_checkpoint_tag


def policy_row(kind, curriculum, encoder_seed, policy_seed, success):
    return {
        "kind": kind, "curriculum": curriculum, "encoder_seed": encoder_seed,
        "policy_seed": policy_seed, "source_sha256": "abc",
        "final": {"success": success, "spl": success / 2,
                  "progress": success / 3},
        "success_auc": success / 4,
    }


def test_parse_checkpoint_tag_preserves_curriculum_underscores():
    assert parse_checkpoint_tag("gru_deprived_early_s2") == (
        "gru", "deprived_early", 2)
    assert parse_checkpoint_tag("rnn_slow_first_s0") == ("rnn", "slow_first", 0)


def test_aggregate_policy_seeds_before_encoder_inference():
    rows = [policy_row("gru", "slow_first", 0, p, .4 + .1 * p)
            for p in (0, 1, 2)]
    out = aggregate(rows, [0, 1, 2])
    assert len(out) == 1
    assert out[0]["n_policy_seeds"] == 3
    assert out[0]["final_success"] == pytest.approx(.5)


def test_aggregate_rejects_unbalanced_policy_seeds():
    rows = [policy_row("gru", "slow_first", 0, 0, .5)]
    with pytest.raises(ValueError, match="policy seeds"):
        aggregate(rows, [0, 1])


def test_paired_test_uses_encoder_units():
    rows = []
    for kind in ("gru", "rnn"):
        for seed in range(4):
            rows.append(aggregate([policy_row(kind, "slow_first", seed, 0, .7)], [0])[0])
            rows.append(aggregate([policy_row(kind, "fast_first", seed, 0, .4)], [0])[0])
    result = paired(rows, "slow_first", "fast_first", "final_success")
    assert result["n_pairs"] == 8
    assert result["exact_p"] == pytest.approx(.0078125)
    assert result["status"] == "PASSABLE"


def test_matrix_validator_reports_missing_cells():
    one = aggregate([policy_row("gru", "slow_first", 0, 0, .5)], [0])
    missing = validate_matrix(one)
    assert ("gru", "slow_first", 0) not in missing
    assert ("rnn", "shuffled", 3) in missing


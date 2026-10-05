"""CPU arithmetic tests only, not model/credit quality evidence or a gate."""

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).with_name("analyze_textcraft_native_minibatch.py")
SPEC = importlib.util.spec_from_file_location("native_minibatch_analyzer", SOURCE)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def test_masked_original_ids_and_signed_mass():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([[248068, 271, 7, 99], [248069, 271, 7, 99]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    dt = torch.tensor([[-2., 1., 0., 1000.], [-3., -1., 4., -1000.]])
    grpo = torch.tensor([[1., 1., 1., 1000.], [-2., -2., -2., -1000.]])
    result = ANALYZER.token_id_mass_summary(ids, mask, dt, grpo, torch)
    records = {record["token_id"]: record for record in result["token_id_records"]}
    assert result["action_token_count"] == 6
    assert 99 not in records
    assert records[271]["action_token_count"] == 2
    assert records[271]["dt_actor_advantages"]["signed_mass"] == 0
    assert records[271]["dt_actor_advantages"]["abs_mass"] == 2
    assert records[248068]["dt_actor_advantages"]["negative_abs_mass"] == 2
    assert records[248069]["dt_actor_advantages"]["negative_abs_mass"] == 3
    assert result["finite_mass_totals"]["dt_actor_advantages"]["signed_mass"] == -1
    assert result["finite_mass_totals"]["dt_actor_advantages"]["abs_mass"] == 11
    assert result["finite_mass_totals"]["official_grpo_advantages"]["signed_mass"] == -3
    assert result["finite_mass_totals"]["official_grpo_advantages"]["abs_mass"] == 9
    assert result["union_mass_coverage"]["dt_actor_advantages"]["negative_abs_mass"]["coverage"] == 1


def test_top20_bound_and_mass_coverage():
    torch = pytest.importorskip("torch")
    ids = torch.arange(24).reshape(1, 24)
    values = torch.arange(1., 25.).reshape(1, 24)
    result = ANALYZER.token_id_mass_summary(ids, torch.ones_like(ids), values, -values, torch)
    ranking = result["top20_rankings"]["dt_actor_advantages"]["positive_abs_mass"]
    assert ranking["token_ids"] == list(range(23, 3, -1))
    assert ranking["captured_mass"] == 290
    assert ranking["coverage"] == 290 / 300
    assert len(result["token_id_records"]) == 23  # 20 ranked + three requested, absent IDs
    assert result["selected_id_action_token_count"] == 20


def test_empty_mask_keeps_requested_ids_without_mass():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([[7]])
    values = torch.tensor([[1000.]])
    result = ANALYZER.token_id_mass_summary(ids, torch.zeros_like(ids), values, values, torch)
    assert result["action_token_count"] == 0
    assert [record["token_id"] for record in result["token_id_records"]] == [271, 248068, 248069]
    assert all(record["action_token_count"] == 0 for record in result["token_id_records"])
    assert result["top20_rankings"]["dt_actor_advantages"]["positive_abs_mass"]["coverage"] is None


def test_nonfinite_values_are_counted_and_excluded_from_mass():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([[248068, 248069, 271]])
    dt = torch.tensor([[float("nan"), float("inf"), -2.]])
    grpo = torch.tensor([[0., 1., -1.]])
    result = ANALYZER.token_id_mass_summary(ids, torch.ones_like(ids), dt, grpo, torch)
    totals = result["finite_mass_totals"]["dt_actor_advantages"]
    assert totals["nonfinite_count"] == 2
    assert totals["finite_count"] == 1
    assert totals["negative_abs_mass"] == 2
    assert totals["signed_mass"] == -2
    records = {record["token_id"]: record for record in result["token_id_records"]}
    assert records[248068]["dt_actor_advantages"]["nonfinite_count"] == 1
    assert records[248069]["official_grpo_advantages"]["positive_abs_mass"] == 1

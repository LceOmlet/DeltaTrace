"""Test the generated DT trainer seam against untouched pinned owner helpers.

Run with ``python -m unittest discover -s tests -p test_dt_official_whitening.py -v``.
Without Torch, only the source/migration tests run. With CPU Torch, the tests
compile the saved owner's original helper definitions and the generated actual
``compute_advantage`` definition; they do not reproduce a whitening formula or
import the full trainer/model stack. Environment variables below can point to
the same frozen source bytes when running this file in the native CPU runtime.
"""

from __future__ import annotations

import ast
from enum import Enum
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "research/temporary/rl_upstream_alignment_20260929"
OWNER = AUDIT / "recipe-sources/verl-agent-20bd331"
sys.path.insert(0, str(ROOT / "experiments/rl"))
import patch_verl_agent2 as patch


SOURCE_PATHS = {
    "pristine": Path(os.environ.get(
        "DT_VERL_PRISTINE_TRAINER_SOURCE", str(OWNER / "verl/trainer/ppo/ray_trainer.py"))),
    "actual": Path(os.environ.get(
        "DT_VERL_TRAINER_SOURCE", str(AUDIT / "textcraft-degradation-20261005/native-minibatch-v4/actual-trainer.py"))),
    "functional": Path(os.environ.get(
        "DT_VERL_TORCH_FUNCTIONAL_SOURCE", str(OWNER / "verl/utils/torch_functional.py"))),
}
SOURCE_SHAS = {
    "pristine": "69dab6ef8a0521704468c8158c6b19c782d5455da49a3a1582d4bcd6a2a51e30",
    "actual": "8816ea4e5f9a95a1dfe1067349eee9101da8bf4ec916bcc28a5b03288976f0df",
    "functional": "079a6d20696a861340687d6e61fb4162cc1d436837ce747ac9ef318b738c1701",
}


def owner_source(name):
    raw = SOURCE_PATHS[name].read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHAS[name]:
        raise AssertionError(f"not the frozen {name} owner source: {SOURCE_PATHS[name]}")
    # Match the patch tool's Path.read_text universal-newline source contract.
    return SOURCE_PATHS[name].read_text(encoding="utf-8")


def definition(source, name):
    return next(node for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def estimator_branches(source):
    function = definition(source, "compute_advantage")
    branch = next(node for node in function.body if isinstance(node, ast.If)
                  and "AdvantageEstimator.GAE" in ast.unparse(node.test))
    result = {}
    while isinstance(branch, ast.If):
        result[ast.unparse(branch.test)] = ast.dump(ast.Module(body=branch.body, type_ignores=[]))
        branch = branch.orelse[0] if len(branch.orelse) == 1 else None
    return result


class SourceContractTests(unittest.TestCase):
    def test_frozen_sources_and_raw_migration_are_idempotent(self):
        actual = owner_source("actual")
        self.assertIn(patch.RAY_ADV_INSERT_RAW_PREVIOUS, actual)
        self.assertNotIn("masked_whiten", patch.RAY_ADV_INSERT_RAW_PREVIOUS)
        candidate = patch.patch_dt_advantage_preprocessing(actual)
        self.assertIn(patch.RAY_ADV_INSERT, candidate)
        self.assertNotIn(patch.RAY_ADV_INSERT_RAW_PREVIOUS, candidate)
        self.assertEqual(candidate, patch.patch_dt_advantage_preprocessing(candidate))
        self.assertEqual(candidate.count(patch.RAY_WHITEN_IMPORT), 1)
        ast.parse(candidate)

    def test_pristine_and_previous_branches_upgrade_without_changing_official_branches(self):
        pristine = owner_source("pristine")
        sources = [pristine, owner_source("actual")]
        for previous in (patch.RAY_ADV_INSERT_PREVIOUS, patch.RAY_ADV_INSERT_STALE):
            source = pristine.replace(patch.RAY_ADV_INSERT_ANCHOR, previous, 1)
            enum = patch.RAY_ADV_ENUM_STALE if previous == patch.RAY_ADV_INSERT_STALE else patch.RAY_ADV_ENUM_NEW
            sources.append(source.replace(patch.RAY_ADV_ENUM_OLD, enum, 1))
        for source in sources:
            with self.subTest(source_has_dt=patch.RAY_ADV_ENUM_NEW in source):
                candidate = patch.patch_dt_advantage_preprocessing(source)
                before, after = estimator_branches(source), estimator_branches(candidate)
                for key, body in before.items():
                    if "DELTATRACE" not in key and "COUNTERFACTUAL" not in key:
                        self.assertEqual(body, after[key], key)
                self.assertEqual(candidate, patch.patch_dt_advantage_preprocessing(candidate))

    def test_dt_branch_calls_owner_once_without_gae_or_raw_reassignment(self):
        candidate = patch.patch_dt_advantage_preprocessing(owner_source("actual"))
        function = definition(candidate, "compute_advantage")
        branch = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "adv_estimator == AdvantageEstimator.DELTATRACE")
        calls = [node for statement in branch.body for node in ast.walk(statement)
                 if isinstance(node, ast.Call)]
        whiten = [node for node in calls if ast.unparse(node.func) == "verl_F.masked_whiten"]
        self.assertEqual(len(whiten), 1)
        self.assertEqual([ast.unparse(arg) for arg in whiten[0].args],
                         ["data.batch['dt_token_advantages']", "response_mask"])
        self.assertEqual(whiten[0].keywords, [])  # Official shift_mean default.
        self.assertFalse(any("core_algos" in ast.unparse(node.func) for node in calls))
        assignments = {ast.unparse(node.targets[0]): ast.unparse(node.value)
                       for statement in branch.body for node in ast.walk(statement)
                       if isinstance(node, ast.Assign)}
        self.assertEqual(assignments["data.batch['advantages']"],
                         "torch.where(response_mask, advantages, 0.0)")
        self.assertEqual(assignments["data.batch['returns']"], "data.batch['dt_q_estimates']")
        self.assertNotIn("masked_whiten", assignments["data.batch[name]"])

    def test_main_uses_the_isolated_helper(self):
        main = definition(Path(patch.__file__).read_text(encoding="utf-8"), "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "patch_dt_advantage_preprocessing"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[0]), "ray_trainer.read_text()")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "CPU Torch unavailable; generated owner execution not run")
class ActualOwnerExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        # These are untouched definitions from the SHA-bound official file.
        functional_source = owner_source("functional")
        functional_ast = ast.Module(
            body=[definition(functional_source, name)
                  for name in ("masked_mean", "masked_var", "masked_whiten")],
            type_ignores=[],
        )
        functional_globals = {"torch": torch}
        exec(compile(ast.fix_missing_locations(functional_ast),
                     str(SOURCE_PATHS["functional"]), "exec"), functional_globals)
        cls.functional = SimpleNamespace(**{name: functional_globals[name]
            for name in ("masked_mean", "masked_var", "masked_whiten")})
        generated = patch.patch_dt_advantage_preprocessing(owner_source("actual"))
        trainer_ast = ast.Module(body=[definition(generated, name) for name in
            ("AdvantageEstimator", "compute_response_mask", "compute_advantage")], type_ignores=[])
        cls.namespace = {"torch": torch, "Enum": Enum, "DataProto": SimpleNamespace,
                         "verl_F": cls.functional}
        exec(compile(ast.fix_missing_locations(trainer_ast),
                     str(SOURCE_PATHS["actual"]), "exec"), cls.namespace)

    def make_data(self, raw=None):
        torch = self.torch
        action_mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 0, 0]], dtype=torch.bool)
        if raw is None:
            raw = torch.tensor([[0.1, 0.3, 0, -0.05, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
        q = torch.tensor([[1, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
        batch = {"responses": torch.arange(10).reshape(2, 5),
                 "attention_mask": torch.ones((2, 7), dtype=torch.long),
                 "loss_mask": torch.cat((torch.zeros((2, 2), dtype=torch.bool), action_mask), dim=1),
                 "dt_token_advantages": raw, "dt_q_estimates": q,
                 "dt_v_estimates": q - raw,
                 "old_log_probs": torch.full((2, 5), -2.0),
                 "ref_log_prob": torch.full((2, 5), -2.5)}
        return SimpleNamespace(batch=batch, non_tensor_batch={}, meta_info={}), action_mask

    def test_full_action_mask_one_official_call_preserves_raw_values(self):
        torch = self.torch
        data, mask = self.make_data()
        before = {key: value.clone() for key, value in data.batch.items()}
        actual_whiten = self.functional.masked_whiten
        calls = []

        def record_owner_call(values, actual_mask):
            calls.append((values.clone(), actual_mask.clone()))
            return actual_whiten(values, actual_mask)

        self.namespace["verl_F"] = SimpleNamespace(masked_whiten=record_owner_call)
        try:
            returned = self.namespace["compute_advantage"](
                data, self.namespace["AdvantageEstimator"].DELTATRACE, multi_turn=True)
        finally:
            self.namespace["verl_F"] = self.functional
        expected = torch.where(mask, actual_whiten(before["dt_token_advantages"], mask), 0.0)
        self.assertIs(returned, data)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0][0], before["dt_token_advantages"]))
        self.assertTrue(torch.equal(calls[0][1], mask))
        self.assertTrue(torch.equal(data.batch["advantages"], expected))
        self.assertTrue(torch.equal(data.batch["response_mask"], mask))
        self.assertTrue(torch.equal(data.batch["returns"], before["dt_q_estimates"]))
        for key, value in before.items():
            self.assertTrue(torch.equal(data.batch[key], value), key)
        self.assertTrue(torch.equal(data.batch["advantages"][~mask], torch.zeros_like(expected[~mask])))
        # Centering changes G0's actor coefficient, while its raw Q/V/A stay zero.
        self.assertTrue(torch.count_nonzero(data.batch["advantages"][1][mask[1]]).item() > 0)
        self.assertTrue(torch.isfinite(data.batch["advantages"]).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_constant_batch_and_owner_small_mask_error_are_unmodified(self):
        torch = self.torch
        data, mask = self.make_data(torch.zeros((2, 5), dtype=torch.float32))
        self.namespace["compute_advantage"](data, self.namespace["AdvantageEstimator"].DELTATRACE)
        self.assertTrue(torch.equal(data.batch["advantages"], torch.zeros((2, 5))))
        for count in (0, 1):
            with self.subTest(valid_action_tokens=count):
                data, _ = self.make_data()
                data.batch["response_mask"] = torch.zeros_like(mask)
                if count:
                    data.batch["response_mask"][0, 0] = True
                with self.assertRaises(ValueError):
                    self.namespace["compute_advantage"](
                        data, self.namespace["AdvantageEstimator"].DELTATRACE)


if __name__ == "__main__":
    unittest.main()

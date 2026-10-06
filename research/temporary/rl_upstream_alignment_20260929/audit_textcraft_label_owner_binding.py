"""Source/asset contract for the original RewardAlphabet.labels diagnostic seam.

Uses the exact frozen owner class AST and already saved official-tokenizer IDs.
The text recorder is not a tokenizer implementation or a model validation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import string
import subprocess
from typing import Any


AUDIT = Path(__file__).resolve().parent
DEG = AUDIT / "textcraft-degradation-20261005"
NATIVE = DEG / "readout-quality-20261006/v2"
LABEL = DEG / "label-encoding-20261006/v2"
OWNER_COMMIT = "99fb5c28d9be064bab22dc8ad949494b3fd85150"
OWNER_SHA = "228afbc7a10841d482c3d73def59dfe9ef192c057a97d76dca57369502a10137"


def source(path):
    b = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class RecordTextOnly:
    """Record original encoder argument text; intentionally do not tokenize."""
    def __init__(self):
        self.texts = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.texts.append(text)
        return []


def main():
    raw = subprocess.check_output(["git", "show", OWNER_COMMIT + ":experiments/rl/reward_readout.py"])
    assert hashlib.sha256(raw).hexdigest() == OWNER_SHA
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardAlphabet")
    ns = {"dataclass": dataclass, "Any": Any, "math": math, "string": string, "json": json}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<frozen-owner-RewardAlphabet>", "exec"), ns)
    owner = ns["RewardAlphabet"]
    observer_path = AUDIT / "verify_textcraft_label_encoding.py"
    observer_tree = ast.parse(observer_path.read_bytes())
    swapped_cls = next(n for n in observer_tree.body if isinstance(n, ast.ClassDef) and n.name == "ReversedLabels")
    exec(compile(ast.Module(body=[swapped_cls], type_ignores=[]), str(observer_path), "exec"), ns)
    original_labels = owner.labels
    replacement = ns["ReversedLabels"].labels
    inputs = load(NATIVE / "original-first-response-cases.json")
    decoded = {c["traj_uid"]: c for c in load(NATIVE / "native-prefix-decoding.json")["cases"]}
    label_raw = [load(LABEL / f"rank{r}-readout.json") for r in (0, 1)]
    observed = {c["traj_uid"]: c for rank in label_raw for c in rank["cases"]}
    inspection = load(LABEL / "native-owner-inspection.json")
    alphabet = owner.for_task("TextCraft", inputs["max_steps"])
    assert alphabet.values == (0.0, 1.0) and alphabet.labels() == "01"
    original_state = (alphabet.task, alphabet.values, alphabet.meanings, alphabet.invalid_action_penalty_coef)
    records = []
    for case in [c for rank in inputs["rank_cases"] for c in rank]:
        rc, dc = observed[case["traj_uid"]], decoded[case["traj_uid"]]
        args = dict(current_step=case["source_step"], max_steps=inputs["max_steps"], sampling=inputs["sampling"])
        token_ids = case["selected_input_ids"]
        assert rc["original_query_ids"] == token_ids[case["source_end"]:-1]
        original_recorder, subclass_recorder, binding_recorder = RecordTextOnly(), RecordTextOnly(), RecordTextOnly()
        alphabet.query_ids(original_recorder, **args)
        ns["ReversedLabels"](*original_state).query_ids(subclass_recorder, **args)
        try:
            owner.labels = replacement
            assert alphabet.labels() == "10"
            event_index = alphabet.observed_index(case["observed_return"])
            alphabet.query_ids(binding_recorder, **args)
            assert original_state == (alphabet.task, alphabet.values, alphabet.meanings, alphabet.invalid_action_penalty_coef)
            assert binding_recorder.texts == subclass_recorder.texts
        finally:
            owner.labels = original_labels
        assert owner.labels is original_labels and alphabet.labels() == "01"
        assert original_recorder.texts == [dc["query_decode"]]
        assert event_index == case["observed_class_index"]
        changed = [i for i, (a, b) in enumerate(zip(rc["original_query_ids"], rc["swapped_query_ids"])) if a != b]
        assert len(rc["original_query_ids"]) == len(rc["swapped_query_ids"]) and len(changed) == 2
        assert sorted((rc["original_query_ids"][i], rc["swapped_query_ids"][i]) for i in changed) == [(15, 16), (16, 15)]
        assert token_ids[-1] == [15, 16][event_index]
        records.append({"traj_uid": case["traj_uid"], "source_step": case["source_step"], "G": case["observed_return"],
                        "unchanged_event_index": event_index, "original_target_ID": [15, 16][event_index],
                        "swapped_target_ID_from_saved_official_label_encoding": [16, 15][event_index],
                        "query_tokens": len(rc["original_query_ids"]), "changed_query_positions": changed,
                        "class_binding_query_text_equals_existing_subclass": True,
                        "original_query_text_equals_saved_official_decode": True,
                        "current_action_IDs_preserved": True})
    assert len(records) == 64 and inspection["checked_cases"] == 64
    assert inspection["original_queries_exact"] and inspection["swapped_queries_equal_length"] and inspection["only_two_label_ids_changed"]
    methods = {n.name: {"start": n.lineno, "end": n.end_lineno} for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    tests = AUDIT.parents[2] / "experiments/rl/test_reward_readout.py"
    report = {"scope": "Source and previously saved official-tokenizer asset contract only; no new tokenizer/model/DT/PG call.",
              "analyzer": source(Path(__file__)),
              "frozen_owner": {"git_commit": OWNER_COMMIT, "path": "experiments/rl/reward_readout.py", "sha256": OWNER_SHA, "bytes": len(raw), "RewardAlphabet_methods": methods},
              "sources": {"existing_label_observer": source(observer_path), "original_cases": source(NATIVE / "original-first-response-cases.json"),
                          "decoded_prefixes": source(NATIVE / "native-prefix-decoding.json"), "label_inspection": source(LABEL / "native-owner-inspection.json"),
                          "existing_owner_tests": source(tests),
                          **{f"label_rank{r}": source(LABEL / f"rank{r}-readout.json") for r in (0, 1)}},
              "existing_official_tokenizer_verification": inspection,
              "contract": {"actual_saved_cases": 64, "query_lengths": sorted({r["query_tokens"] for r in records}),
                           "query_changed_positions": sorted({tuple(r["changed_query_positions"]) for r in records}),
                           "original_event_order_values": [0.0, 1.0], "original_label_IDs_event_order": [15, 16], "swapped_label_IDs_event_order": [16, 15],
                           "observed_index_is_semantic_value_index_and_unchanged": True,
                           "label_ids_and_query_ids_both_call_original_self_labels": True,
                           "existing_instance_resolves_temporarily_bound_class_method": True,
                           "original_labels_method_restored": owner.labels is original_labels,
                           "original_query_and_observed_index_methods_not_replaced": True,
                           "source_action_prompt_reward_horizon_sampling_unchanged": True,
                           "original_readout_target_construction": "labels = alphabet.label_ids(tokenizer); observed=alphabet.observed_index(value); target=labels[observed] (owner185/207/214)",
                           "native_paired_layout_mapping": "Completed native reader scores physical IDs[15,16]; swapped success uses column0. A recomputed owner readout obtains event-ordered IDs[16,15] and unchanged success index1, therefore the same target ID15."},
              "rows": records,
              "existing_test_scope": "test_reward_readout.py exercises original label/observed-index/query/readout APIs, but is not a previously executed temporary-label-DI or real-tokenizer test; saved label v2 CPU inspection supplies the real encoding evidence.",
              "operations": {"original_class_AST_methods_called_for_text": True, "official_tokenizer_new_calls": 0, "model_calls": 0, "DT_calls": 0, "backward_calls": 0, "updates": 0, "production_changes": 0},
              "limits": ["RecordTextOnly only records the argument produced by the original query encoder; actual token-ID evidence is reused from completed official-tokenizer CPU inspection and native job, not newly simulated.",
                         "The 64 saved first responses all use current_step0. Runtime original class dispatch also applies to later steps, but this receipt does not claim new full192-request recomputation.",
                         "Recomputed DT/PG diagnostic must bind the exact imported old228afbc7 class, generate fresh query/target through its original methods, restore labels finally, and preserve original collective/worker lifecycle.",
                         "This establishes a legitimate diagnostic-only owner seam and event identity; it does not establish stronger credit, improved gradients, official numerical tolerance, or training recovery."]}
    output = LABEL / "label-owner-binding-contract.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": source(output), "analyzer": report["analyzer"], "cases": len(records), "scope": report["scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

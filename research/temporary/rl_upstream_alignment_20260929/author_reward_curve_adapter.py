"""Literal-ID and categorical-score adapter for the unchanged author metric.

Research only: the author owns ordering, cumulative deletion, density and
RISE/MAS.  The caller owns the native reader and its precision/lifecycle.
Saved d is the diagnostic inverse of stored FP32 A, not a fresh DT vector.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace


_PROMPT = "<opaque-literal-reward-prefix>"
_FORMATTED = "<opaque-formatted-literal-reward-prefix>"
_TARGET = "<opaque-literal-observed-reward-target>"
_FIELDS = ("scores", "density", "normalized_model_response",
           "alignment_penalty", "corrected_scores")


def _ids_sha256(ids):
    return hashlib.sha256(struct.pack("<" + "q" * len(ids), *ids)).hexdigest()


def _source(obj):
    path = Path(inspect.getsourcefile(obj)).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "qualname": obj.__qualname__, "line": inspect.getsourcelines(obj)[1]}


def owner_sources():
    """Inspect the actual imported owner; do not construct a model or evaluator."""
    author = importlib.import_module("ft_ifr_improve")
    evaluator = importlib.import_module("llm_attr_eval")
    result = {"faithfulness_test_skip_tokens": _source(author.faithfulness_test_skip_tokens)}
    for name in ("format_prompt", "compute_logprob_response_given_prompt",
                 "_ensure_pad_token_id"):
        result[name] = _source(getattr(evaluator.LLMAttributionEvaluator, name))
    for name in ("llm_attr", "ifr_core"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        result[name] = {"path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def load_saved_first_response_cases(path):
    """Read the original document without reconstructing or grouping its cases."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                             "bytes": len(raw)}


def inspect_case_contract(case, tokenizer, eos_token_id):
    """CPU/stdlib check of the actual saved carrier; no tokenizer/model call."""
    ids = [int(i) for i in case["selected_input_ids"]]
    start, end = int(case["source_start"]), int(case["source_end"])
    labels = [int(i) for i in case["outcome_token_ids"]]
    observed = int(case["observed_class_index"])
    target = int(case["target_id"])
    if not (0 <= start < end <= len(ids) - 1):
        raise ValueError("Saved source span is outside the literal predictor prefix")
    if not (0 <= observed < len(labels)) or target != labels[observed] or ids[-1] != target:
        raise ValueError("Saved observed target and categorical label IDs disagree")
    if int(tokenizer.eos_token_id) != int(eos_token_id):
        raise ValueError("Deletion EOS differs from the actual tokenizer EOS")
    saved = case.get("saved_source_d_from_A")
    if saved is not None and len(saved) != end - start:
        raise ValueError("Saved d and literal source span lengths disagree")
    return {
        "traj_uid": case["traj_uid"], "source_step": int(case["source_step"]),
        "observed_return": float(case["observed_return"]),
        "observed_class_index": observed, "outcome_token_ids": labels,
        "target_ids": [target], "source_start": start, "source_end": end,
        "source_tokens": end - start, "prefix_tokens": len(ids) - 1,
        "literal_input_sha256": _ids_sha256(ids),
        "literal_prefix_sha256": _ids_sha256(ids[:-1]),
        "literal_target_sha256": _ids_sha256([target]),
        "eos_token_id": int(eos_token_id), "target_eos_appended": False,
        "saved_d_available": saved is not None,
        "saved_d_scope": case.get("saved_credit_diagnostic", {}).get(
            "inverse_scope", case.get("d_missing_scope")),
        "author_k": 20, "author_steps": min(20, end - start),
        "score_points_per_view": min(20, end - start) + 1,
    }


class _LiteralIdsProvider:
    """Supply two exact owner artifacts through the author's tokenizer seam."""
    def __init__(self, torch, prefix, target, eos_token):
        self.prefix = torch.tensor([prefix], dtype=torch.long)
        self.target = torch.tensor([[target]], dtype=torch.long)
        self.eos_token = eos_token

    def __call__(self, text, *, return_tensors, add_special_tokens):
        if return_tensors != "pt" or add_special_tokens is not False:
            raise ValueError("Unexpected author literal-carrier call")
        if text == _FORMATTED:
            return SimpleNamespace(input_ids=self.prefix)
        if text == _TARGET + self.eos_token:
            # The accepted event target is one label, not label plus text EOS.
            return SimpleNamespace(input_ids=self.target)
        raise ValueError("Unexpected opaque marker; no text is re-tokenized")


class _LiteralScoreEvaluator:
    def __init__(self, torch, case, tokenizer, eos_token_id, score_callback,
                 author_function, saved_d, view, phase_callback):
        self.device = torch.device("cpu")
        self.tokenizer = _LiteralIdsProvider(torch, case["selected_input_ids"][:-1],
                                            int(case["target_id"]), tokenizer.eos_token)
        self.eos_token_id = int(eos_token_id)
        self.score_callback = score_callback
        self.author_function = author_function
        self.saved_d = saved_d
        self.source_start = int(case["source_start"])
        self.view, self.phase_callback = view, phase_callback
        self.points = []

    def format_prompt(self, prompt):
        if prompt != " " + _PROMPT:
            raise ValueError("Unexpected author prompt marker")
        return _FORMATTED

    def _ensure_pad_token_id(self):
        return self.eos_token_id

    def compute_logprob_response_given_prompt(self, prompt_ids, response_ids):
        # Read the author's actual cumulative group, including original EOS
        # positions whose replacement would not appear in an ID diff.
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code is not self.author_function.__code__:
                raise RuntimeError("Score callback did not originate in the author metric")
            deleted = [int(j) for j in frame.f_locals.get("sorted_keep", [])[
                :int(frame.f_locals.get("start", 0))]]
            original_group = [int(j) for j in frame.f_locals.get("group", [])]
        finally:
            del frame
        began = time.perf_counter()
        raw = self.score_callback(prompt_ids, response_ids)
        if tuple(raw.shape) != (1, 1):
            raise ValueError("Native scorer must return only this observed category as [1,1]")
        point = {
            "point_index": len(self.points), "raw_logp": float(raw.detach().cpu().item()),
            "author_deleted_source_indices": deleted,
            "author_current_group_source_indices": original_group,
            "author_deleted_input_positions": [self.source_start + j for j in deleted],
            "same_set_saved_d_sum": math.fsum(self.saved_d[j] for j in deleted),
            "scorer_seconds": time.perf_counter() - began,
        }
        self.points.append(point)
        if self.phase_callback is not None:
            self.phase_callback(self.view, "score_complete", **point)
        return raw


def run_saved_reward_curves(case, tokenizer, score_callback, *, eos_token_id,
                           phase_callback=None):
    """Call the untouched author twice; the callback owns native model scoring.

    ``score_callback(prompt_ids, response_ids)`` receives literal CPU tensors
    and must delegate to the caller's original native reader, select the saved
    observed category, and return a single [1,1] log probability.  It owns
    device transfer, native precision and model lifecycle. No DT is re-run.
    """
    contract = inspect_case_contract(case, tokenizer, eos_token_id)
    if not contract["saved_d_available"]:
        raise ValueError("G0 has no saved d: no zero attribution is fabricated for a curve")
    import torch
    author = importlib.import_module("ft_ifr_improve")
    function = author.faithfulness_test_skip_tokens
    saved_d = [float(v) for v in case["saved_source_d_from_A"]]
    source_d = torch.tensor(saved_d, dtype=torch.float64)
    views = {"positive_MAS": source_d.float().clamp_min(0),
             "signed_RISE": source_d.float()}
    results = {}
    for view, attribution in views.items():
        evaluator = _LiteralScoreEvaluator(torch, case, tokenizer, eos_token_id,
                                          score_callback, function, saved_d, view,
                                          phase_callback)
        captured = {}
        previous = sys.getprofile()

        def capture(frame, event, value):
            if previous is not None:
                previous(frame, event, value)
            if frame.f_code is function.__code__ and event == "return" and value is not None:
                for field in _FIELDS:
                    captured[field] = frame.f_locals[field].tolist()

        if phase_callback is not None:
            phase_callback(view, "curve_begin", score_points=contract["score_points_per_view"])
        started = time.perf_counter()
        sys.setprofile(capture)
        try:
            with torch.no_grad():
                metrics = function(
                    evaluator, attribution[None], _PROMPT, _TARGET,
                    keep_prompt_token_indices=range(len(saved_d)),
                    user_prompt_indices=range(contract["source_start"], contract["source_end"]),
                    k=20)
        finally:
            sys.setprofile(previous)
        results[view] = {
            "author_return": [float(v) for v in metrics],
            "author_return_fields": ["rise", "mas", "rise_plus_ap"],
            "author_curve": captured, "score_points": evaluator.points,
            "native_score_calls": len(evaluator.points),
            "seconds": time.perf_counter() - started,
            "attribution_view": "saved d cast FP32" + (" then clamp_min(0)" if view == "positive_MAS" else ""),
        }
        factual = evaluator.points[0]["raw_logp"]
        for point in evaluator.points:
            point["native_factual_minus_deleted_logp"] = factual - point["raw_logp"]
        if phase_callback is not None:
            phase_callback(view, "curve_complete", seconds=results[view]["seconds"],
                           native_score_calls=len(evaluator.points))
    return {
        "scope": "Unchanged author cumulative deletion/RISE/MAS with literal reward-target scoring; not fresh DT or a tolerance gate",
        "case_contract": contract, "owner_sources": owner_sources(), "views": results,
        # Same field selection as experiments/official/evaluate.py:use_signed_rise;
        # all three original results remain above, with no metric recomputation.
        "metrics": {"rise": results["signed_RISE"]["author_return"][0],
                    "mas": results["positive_MAS"]["author_return"][1],
                    "rise_plus_ap": results["positive_MAS"]["author_return"][2]},
        "saved_d_scope": "FP64 inverse from stored FP32 A; not the pre-storage raw finite vector",
        "native_score_calls": sum(v["native_score_calls"] for v in results.values()),
        "finite_trace_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
    }

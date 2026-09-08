"""Paired layer0 diagnostic for the existing symmetric GDN norm-gate rule.

Consume actual same-run GPU endpoints and the current decoder's coefficients.
Only the GDN pullback and input norm/residual run again. No model forward,
MLP, postnorm, other layer, or alternate FA/FLA implementation is evaluated.
"""
import json
import math
import time

import torch

from qwen35_gdn_finite import gdn_finite_pullback


UNCHANGED_RELATIVE_L2_LIMIT = 1e-6
ENDPOINT_KEYS = frozenset(
    ("raw_g", "q", "k", "v", "g", "beta", "A", "w", "v_new", "o", "h")
)


class SymmetricNormgateDiagnosticError(RuntimeError):
    """Keep JSON-safe evidence available to the experiment's failure saver."""

    def __init__(self, message, metadata):
        self.metadata = metadata
        super().__init__(message + ": " + json.dumps(metadata, sort_keys=True))


def _difference_stats(actual, reference):
    """Describe a difference; only the caller decides whether it is a guard."""
    exact = bool(torch.equal(actual.contiguous().view(torch.uint8),
                             reference.contiguous().view(torch.uint8)))
    delta = actual.double() - reference.double()
    difference_norm = float(delta.norm())
    reference_norm = float(reference.double().norm())
    return {
        "bitwise_equal": exact,
        "difference_L2": difference_norm,
        "reference_L2": reference_norm,
        "relative_L2": difference_norm / reference_norm if reference_norm else None,
        "max_absolute": float(delta.abs().max()),
    }


def compute_layer0_symmetric_normgate(
    layer, d, c, e, upstream, current_new, terms, fla, boundaries, scale
):
    """Return (candidate_new_GPU, candidate_decoder_terms, metadata).

The caller supplies scale captured from the current actual finite FLA call;
it is never inferred from head dimensions. The caller also establishes the
same-run provenance of d/c/e/upstream/current_new/terms. Returned tensors stay
on GPU for the caller's existing CPU observer/export path.
"""
    metadata = {
        "status": "starting",
        "scope": "Layer0 norm_gate_rule content1 -> symmetric; fixed actual paired endpoints.",
        "claims": "Diagnostic candidate only; no bitwise-equivalence, MAS improvement, or speed claim.",
        "provenance_contract": "Caller supplies the same actual run's d/c/e/upstream/current terms and last FLA scale.",
        "calls": [],
        "tensor_checks": {},
        "additional_work": {
            "finite_FLA_calls_entered": 0,
            "finite_FLA_calls_returned": 0,
            "GDN_repropagations_entered": 0,
            "GDN_repropagations_returned": 0,
            "native_FLA_adjoint_stage_calls": 0,
            "public_paired_conv_preactivation_calls": 0,
            "public_conv_autograd_calls": 0,
            "input_norm_residual_calls_entered": 0,
            "input_norm_residual_calls_returned": 0,
            "extra_model_forwards": 0,
            "finite_FA_calls": 0,
            "MLP_repropagations": 0,
            "postnorm_repropagations": 0,
            "other_layer_repropagations": 0,
        },
        "cost_contract": (
            "The actual baseline DT already paid for current GDN/FLA/conv/inputnorm. "
            "This paired diagnostic adds one finite FLA call (two unchanged native "
            "adjoint stages), one GDN propagation including one paired public conv "
            "preactivation and one public conv autograd call, and one input norm/residual. "
            "A normal complete symmetric candidate substitutes that GDN call and uses "
            "the same number of GDN/FLA/conv calls as current DT. Diagnostic duplication "
            "is not a production candidate overhead measurement. Validation and "
            "FP64 endpoint contractions are additional diagnostic work."
        ),
        "count_basis": (
            "FLA/GDN/inputnorm entries and returns are counted here. Two native FLA "
            "adjoints and conv preactivation/autograd counts follow successful returns "
            "of the existing unchanged implementations; partial failed internal work "
            "is unknown, not asserted to be zero."
        ),
        "timing_contract": (
            "Synchronized GPU wall time including host dispatch. FLA timing is nested "
            "inside GDN timing; do not sum nested records. GDN timing includes conv "
            "preactivation/autograd, whose individual latencies are not isolated. "
            "Complete diagnostic time includes guards, comparisons and contractions "
            "after the initial synchronization; excludes caller CPU exports."
        ),
    }
    device = None
    started = None

    def require(condition, message):
        if not condition:
            metadata["status"] = "failed_guard"
            raise SymmetricNormgateDiagnosticError(message, metadata)

    def check_tensor(name, value, expected_shape=None):
        require(isinstance(value, torch.Tensor), "Missing tensor: " + name)
        row = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        metadata["tensor_checks"][name] = row
        require(value.device == device, "Tensor device changed: " + name)
        if expected_shape is not None:
            row["expected_shape"] = list(expected_shape)
            require(tuple(value.shape) == tuple(expected_shape), "Tensor shape changed: " + name)
        row["finite"] = bool(torch.isfinite(value).all())
        require(row["finite"], "Nonfinite tensor: " + name)

    def timed(kind, operation):
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        record = {"kind": kind, "status": "entered", "end_synchronized": False}
        metadata["calls"].append(record)
        try:
            result = operation()
            torch.cuda.synchronize(device)
            record["end_synchronized"] = True
            record["status"] = "returned"
            return result
        except Exception as error:
            record.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            if not record["end_synchronized"]:
                try:
                    torch.cuda.synchronize(device)
                    record["end_synchronized"] = True
                except Exception as error:
                    record["synchronization_error"] = type(error).__name__ + ": " + str(error)
            record["wall_seconds"] = time.perf_counter() - tick

    def contraction(coefficient, endpoints):
        delta = endpoints[1::2].double() - endpoints[0::2].double()
        require(coefficient.shape == delta.shape, "Endpoint contraction shape changed")
        per_pair = (coefficient.double() * delta).flatten(1).sum(1)
        require(bool(torch.isfinite(per_pair).all()), "Nonfinite endpoint contraction")
        return {"total": float(per_pair.sum()), "per_pair": per_pair.detach().cpu().tolist()}

    def normgate_effect(gdn_terms):
        content = contraction(gdn_terms["mo_before_cast"], e["o"])
        gate = contraction(gdn_terms["mz"], c["z"])
        return {
            "content_before_native_cast": content,
            "gate": gate,
            "total": content["total"] + gate["total"],
            "per_pair": [x + y for x, y in zip(content["per_pair"], gate["per_pair"])],
        }

    try:
        require(layer.block_type == "linear_attention", "Require actual GDN decoder")
        require(layer.linear_attn.layer_idx == 0, "Diagnostic supports only layer0")
        require(callable(fla), "Require existing finite FLA callback")
        require(set(e) == ENDPOINT_KEYS, "Unreviewed native endpoint fields")
        scale = float(scale)
        require(math.isfinite(scale) and scale > 0, "Missing finite positive actual FLA scale")
        require(isinstance(upstream, torch.Tensor) and upstream.is_cuda and upstream.ndim == 3,
                "Require actual GPU decoder upstream [B,T,width]")
        device = upstream.device
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        metadata.update(actual_scale=scale, scale_source="Caller-captured actual last FLA call")
        b, t, width = upstream.shape
        require(b > 0 and t > 0 and width > 0, "Empty decoder dimensions")
        paired_shape = (2 * b, t, width)
        module = layer.linear_attn
        value_shape = (2 * b, t, module.num_v_heads, module.head_v_dim)
        key_shape = (2 * b, t, module.num_v_heads, module.head_k_dim)
        mixer_terms = terms["mixer"]
        fixed_mixer_output = terms["m_mixer_output"]

        with torch.no_grad():
            for name, value in (("upstream", upstream), ("current_new", current_new),
                                ("m_mixer_output", fixed_mixer_output),
                                ("m_mixer_input", terms["m_mixer_input"]),
                                ("m_mlp_norm_output", terms["m_mlp_norm_output"])):
                check_tensor(name, value, upstream.shape)
            for name in ("input_norm_input", "input_norm_output", "output"):
                check_tensor("d." + name, d[name], paired_shape)
            for name in ("input", "output"):
                check_tensor("c." + name, c[name], paired_shape)
            check_tensor("c.z", c["z"], value_shape)
            check_tensor("c.norm_output", c["norm_output"], (2 * b, t, value_shape[2] * value_shape[3]))
            for name, value in e.items():
                check_tensor("e." + name, value)
                require(value.ndim > 0 and value.shape[0] == 2 * b, "Unpaired endpoint: " + name)
            for name in ("q", "k", "w"):
                require(tuple(e[name].shape) == key_shape, "Native key shape changed: " + name)
            for name in ("v", "v_new", "o"):
                require(tuple(e[name].shape) == value_shape, "Native value shape changed: " + name)
            for name in ("raw_g", "g", "beta"):
                require(tuple(e[name].shape) == value_shape[:-1], "Native scalar shape changed: " + name)
            check_tensor("current.mnorm", mixer_terms["mnorm"], (b, t, value_shape[2] * value_shape[3]))
            for name in ("mo_before_cast", "mo_native", "mz"):
                check_tensor("current." + name, mixer_terms[name], (b, *value_shape[1:]))

            def same_endpoint_fla(actual_e, actual_do, actual_scale):
                require(actual_e is e, "Candidate changed actual endpoint dictionary")
                require(float(actual_scale) == scale, "Candidate changed actual FLA scale")
                check_tensor("candidate.FLA_upstream", actual_do, mixer_terms["mo_native"].shape)
                require(actual_do.dtype == mixer_terms["mo_native"].dtype, "Native FLA upstream dtype changed")
                work = metadata["additional_work"]
                work["finite_FLA_calls_entered"] += 1
                require(work["finite_FLA_calls_entered"] == 1, "Unexpected extra finite FLA call")
                result = timed("symmetric_finite_FLA_nested_in_GDN", lambda: fla(actual_e, actual_do, actual_scale))
                work["finite_FLA_calls_returned"] += 1
                work["native_FLA_adjoint_stage_calls"] += 2
                return result

            work = metadata["additional_work"]
            work["GDN_repropagations_entered"] += 1
            new_gdn, new_gdn_terms = timed(
                "symmetric_GDN_including_FLA_and_conv_preactivation_autograd",
                lambda: gdn_finite_pullback(
                    layer.linear_attn, c, e, fixed_mixer_output, scale,
                    same_endpoint_fla, diagnostics=True, norm_gate_rule="symmetric",
                ),
            )
            work["GDN_repropagations_returned"] += 1
            work["public_paired_conv_preactivation_calls"] += 1
            work["public_conv_autograd_calls"] += 1
            require(work["finite_FLA_calls_returned"] == 1, "Candidate did not call finite FLA exactly once")
            check_tensor("candidate.m_mixer_input", new_gdn, terms["m_mixer_input"].shape)
            require(set(new_gdn_terms) == set(mixer_terms), "Candidate GDN diagnostic fields changed")
            for name, value in new_gdn_terms.items():
                if isinstance(value, torch.Tensor):
                    check_tensor("candidate." + name, value, mixer_terms[name].shape)
                    require(value.dtype == mixer_terms[name].dtype, "Candidate coefficient dtype changed: " + name)
            mnorm_check = _difference_stats(new_gdn_terms["mnorm"], mixer_terms["mnorm"])
            mnorm_check["relative_L2_limit"] = UNCHANGED_RELATIVE_L2_LIMIT
            mnorm_check["passed"] = mnorm_check["bitwise_equal"] or (
                mnorm_check["relative_L2"] is not None
                and mnorm_check["relative_L2"] <= UNCHANGED_RELATIVE_L2_LIMIT
            )
            metadata["mnorm_check"] = mnorm_check
            require(mnorm_check["passed"], "Fixed mnorm coefficient drifted")
            metadata["expected_changed_coefficients"] = {
                name: _difference_stats(new_gdn_terms[name], mixer_terms[name])
                for name in ("mo_before_cast", "mo_native", "mz")
            }
            require(set(new_gdn_terms["coeff"]) == set(mixer_terms["coeff"]), "FLA coefficient fields changed")
            for name, value in new_gdn_terms["coeff"].items():
                check_tensor("candidate.coeff." + name, value, mixer_terms["coeff"][name].shape)
            work["input_norm_residual_calls_entered"] += 1
            candidate_new = timed(
                "input_norm_residual_with_fixed_original_residual",
                lambda: boundaries.norm_residual(
                    d["input_norm_input"][0::2], d["input_norm_input"][1::2],
                    layer.input_layernorm.weight, new_gdn,
                    fixed_mixer_output, layer.input_layernorm.eps,
                ),
            )
            work["input_norm_residual_calls_returned"] += 1
            check_tensor("candidate_new", candidate_new, current_new.shape)
            require(candidate_new.dtype == current_new.dtype, "Candidate decoder coefficient dtype changed")
            candidate_terms = dict(terms, m_mixer_input=new_gdn, mixer=new_gdn_terms)
            metadata["fixed_coefficient_reuse"] = {
                "m_mixer_output_same_object": candidate_terms["m_mixer_output"] is fixed_mixer_output,
                "m_mlp_norm_output_same_object": candidate_terms["m_mlp_norm_output"] is terms["m_mlp_norm_output"],
                "residual_source": "Original terms['m_mixer_output'] reused directly",
            }
            metadata["B2_endpoint_contractions"] = timed(
                "actual_paired_endpoint_contractions",
                lambda: {
                    "orientation": "row1-minus-row0 for each original paired sample; actual effects are fixed-coefficient contractions of native captured outputs",
                    "normgate": {
                        "actual_effect": contraction(mixer_terms["mnorm"], c["norm_output"]),
                        "current": normgate_effect(mixer_terms),
                        "candidate": normgate_effect(new_gdn_terms),
                    },
                    "GDN": {
                        "actual_effect": contraction(fixed_mixer_output, c["output"]),
                        "current": contraction(terms["m_mixer_input"], c["input"]),
                        "candidate": contraction(new_gdn, c["input"]),
                    },
                    "decoder": {
                        "actual_effect": contraction(upstream, d["output"]),
                        "current": contraction(current_new, d["input_norm_input"]),
                        "candidate": contraction(candidate_new, d["input_norm_input"]),
                    },
                },
            )
            metadata["status"] = "layer0_symmetric_normgate_computed"
            return candidate_new, candidate_terms, metadata
    except SymmetricNormgateDiagnosticError:
        raise
    except Exception as error:
        metadata["status"] = "failed_execution"
        metadata["execution_error"] = type(error).__name__ + ": " + str(error)
        raise SymmetricNormgateDiagnosticError("Symmetric norm-gate diagnostic failed", metadata) from error
    finally:
        if started is not None:
            try:
                torch.cuda.synchronize(device)
            except Exception as error:
                metadata["final_synchronization_error"] = type(error).__name__ + ": " + str(error)
            metadata["complete_diagnostic_seconds"] = time.perf_counter() - started

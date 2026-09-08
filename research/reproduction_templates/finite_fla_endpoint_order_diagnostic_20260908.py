"""Bounded layer0 endpoint-order diagnostic using the existing finite FLA.

This never changes a model forward, native backward, or the finite kernel.
Average the complete forward/reverse endpoint allocations, not each local
product rule. The caller supplies actual current-run GPU captures and currents.
No original RISE/MAS or complete candidate speed claim is made by this helper.
"""
import json
import math
import time

import torch

from qwen35_gdn_finite import gdn_finite_pullback


ENDPOINT_KEYS = frozenset(
    ("raw_g", "q", "k", "v", "g", "beta", "A", "w", "v_new", "o", "h")
)
COEFFICIENT_KEYS = ("q", "k", "v", "beta", "alpha", "g")
PROPAGATED_KEYS = ("q", "k", "v", "beta", "g")
UNCHANGED_RELATIVE_L2_LIMIT = 1e-6


class EndpointOrderDiagnosticError(RuntimeError):
    """Preserve diagnostic evidence when a frozen structural guard fails."""

    def __init__(self, message, metadata):
        self.metadata = metadata
        super().__init__(message + ": " + json.dumps(metadata, sort_keys=True))


class CountedFLA:
    """Delegate unchanged finite FLA calls; retain metadata, never tensors."""

    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self.last_endpoints_id = None
        self.last_scale = None
        self.last_do_id = None

    def __call__(self, endpoints, do, scale, kind="current"):
        self.last_endpoints_id = id(endpoints)
        self.last_scale = float(scale)
        self.last_do_id = id(do)
        record = {
            "kind": kind,
            "endpoints_id": id(endpoints),
            "scale": float(scale),
            "do_shape": list(do.shape),
            "do_dtype": str(do.dtype),
            "status": "entered",
        }
        self.calls.append(record)
        started = time.perf_counter()
        try:
            output = self.backend(endpoints, do, scale)
            record["status"] = "returned"
            return output
        except Exception as error:
            record["status"] = "failed"
            record["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            # No synchronization is introduced into each current DT call.
            # This is host dispatch time, not device execution latency.
            record["host_dispatch_seconds_not_GPU_latency"] = (
                time.perf_counter() - started
            )


def _difference_stats(actual, reference):
    stats = {
        "shape": list(actual.shape),
        "reference_shape": list(reference.shape),
        "dtype": str(actual.dtype),
        "reference_dtype": str(reference.dtype),
        "relative_L2_limit": UNCHANGED_RELATIVE_L2_LIMIT,
    }
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return dict(stats, passed=False, reason="shape_or_dtype_changed")
    exact = bool(torch.equal(actual, reference))
    finite = bool(torch.isfinite(actual).all() & torch.isfinite(reference).all())
    stats.update(bitwise_equal=exact, finite=finite)
    if not finite:
        return dict(stats, passed=False, relative_L2=None, max_absolute=None)
    if exact:
        return dict(stats, passed=True, relative_L2=0.0, max_absolute=0.0)
    delta = actual.double() - reference.double()
    delta_norm = float(delta.norm())
    reference_norm = float(reference.double().norm())
    relative = delta_norm / reference_norm if reference_norm else None
    passed = (
        delta_norm == 0.0
        if reference_norm == 0.0
        else relative <= UNCHANGED_RELATIVE_L2_LIMIT
    )
    return dict(
        stats,
        passed=bool(passed),
        relative_L2=relative,
        max_absolute=float(delta.abs().max()),
        reference_norm=reference_norm,
    )


def compute_layer0_order_average(
    layer, d, c, e, upstream, current_new, terms, fla, boundaries
):
    """Return (candidate_signed_CPU[B,T], reverse_coeff_CPU, metadata).

    Upstream/MLP/postnorm coefficients and all original d/c/e endpoints remain
    fixed. Only FLA input coefficients change; repropagate their average through
    the unchanged GDN surroundings and input norm/residual. The original
    whole-DT call has already paid for current coefficients and its convolution.
    """
    metadata = {
        "status": "starting",
        "scope": "Layer0 complete FLA endpoint-order average; fixed actual upstream.",
        "method_claim": "Diagnostic only, no new-order original RISE/MAS or speed claim.",
        "calls": [],
        "unchanged_coefficient_checks": {},
        "additional_work": {
            "finite_FLA_calls_entered": 0,
            "finite_FLA_calls_returned": 0,
            "GDN_repropagations_entered": 0,
            "GDN_repropagations_returned": 0,
            "cached_average_callback_calls": 0,
            "input_norm_residual_calls": 0,
            "extra_model_forwards": 0,
            "MLP_repropagations": 0,
            "postnorm_repropagations": 0,
        },
        "cost_contract": (
            "A successful diagnostic adds one reverse finite FLA call (two "
            "unchanged native FLA adjoint stages), one GDN repropagation with "
            "one public paired conv preactivation and one public autograd call, "
            "and one input norm/residual. Current FLA and its first conv were "
            "already paid by whole DT. A normally executed averaged candidate "
            "still needs TWO finite FLA calls at layer0, but only one surrounding "
            "GDN/conv propagation. No all-layer candidate speed benchmark."
        ),
    }

    def require(condition, message):
        if not condition:
            metadata["status"] = "failed_guard"
            raise EndpointOrderDiagnosticError(message, metadata)

    require(isinstance(fla, CountedFLA), "Require the counted unchanged backend")
    require(layer.block_type == "linear_attention", "Require the actual GDN decoder")
    require(layer.linear_attn.layer_idx == 0, "This diagnostic is only for layer0")
    require(set(e) == ENDPOINT_KEYS, "Unreviewed endpoint fields")
    require(fla.last_endpoints_id == id(e), "Current coefficients used different endpoints")
    require(bool(fla.calls) and fla.calls[-1]["status"] == "returned",
            "Current finite FLA call did not return")
    scale = fla.last_scale
    require(scale is not None and math.isfinite(scale), "Missing actual finite FLA scale")
    mixer_terms = terms["mixer"]
    fixed_do = mixer_terms["mo_native"]
    require(fla.last_do_id == id(fixed_do), "Current coefficients used a different do tensor")
    require(set(mixer_terms["coeff"]) == set(COEFFICIENT_KEYS), "Unreviewed coefficient keys")
    require(e["q"].is_cuda and fixed_do.is_cuda, "Require actual GPU captures")
    endpoint_rows = e["q"].shape[0]
    require(endpoint_rows > 0 and endpoint_rows % 2 == 0, "Endpoints are not paired")
    require(all(isinstance(x, torch.Tensor) and x.shape[0] == endpoint_rows
                and x.device == e["q"].device for x in e.values()),
            "Endpoint dimensions/devices disagree")
    require(upstream.shape == current_new.shape == terms["m_mixer_output"].shape,
            "Actual layer coefficient dimensions disagree")
    require(d["input_norm_input"].shape[0] == endpoint_rows,
            "Actual layer input endpoint rows disagree")
    device = fixed_do.device
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    before_call_count = len(fla.calls)
    metadata.update(actual_scale=scale, endpoint_fields=sorted(ENDPOINT_KEYS),
                    current_FLA_call_index=before_call_count - 1)

    def timed(kind, operation):
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        value = operation()
        torch.cuda.synchronize(device)
        metadata["calls"].append({"kind": kind, "seconds": time.perf_counter() - tick})
        return value

    def paired_contraction(coefficient, actual_endpoints):
        delta = actual_endpoints[1::2].double() - actual_endpoints[0::2].double()
        require(coefficient.shape == delta.shape, "Contraction coordinates disagree")
        return float((coefficient.double() * delta).sum())

    def coefficient_effect(coefficients):
        branches = {
            name: paired_contraction(coefficients[name], e["raw_g" if name == "g" else name])
            for name in PROPAGATED_KEYS
        }
        return {"branches": branches, "total": sum(branches.values())}

    try:
        with torch.no_grad():
            order = torch.arange(endpoint_rows, device=device).reshape(-1, 2).flip(1).flatten()
            reverse_e = timed(
                "exchange_all_actual_endpoint_rows",
                lambda: {name: value.index_select(0, order) for name, value in e.items()},
            )
            metadata["endpoint_permutation"] = order.detach().cpu().tolist()
            metadata["additional_work"]["finite_FLA_calls_entered"] += 1
            reverse = timed(
                "reverse_endpoint_finite_FLA",
                lambda: fla(reverse_e, fixed_do, scale, kind="reverse"),
            )
            metadata["additional_work"]["finite_FLA_calls_returned"] += 1
            require(set(reverse) == set(COEFFICIENT_KEYS), "Unreviewed reverse coefficient keys")
            current = mixer_terms["coeff"]
            for name in COEFFICIENT_KEYS:
                require(reverse[name].shape == current[name].shape
                        and reverse[name].dtype == current[name].dtype,
                        "Reverse coefficient contract changed: " + name)
                require(bool(torch.isfinite(reverse[name]).all()
                             & torch.isfinite(current[name]).all()),
                        "Nonfinite coefficient: " + name)
            average = timed(
                "average_six_finite_coefficients",
                lambda: {name: (current[name] + reverse[name]) * 0.5
                         for name in COEFFICIENT_KEYS},
            )
            # alpha and g are alternative coordinates of the same decay term;
            # only raw-g coefficients are contracted below. No double counting.
            endpoint_output = paired_contraction(fixed_do, e["o"])
            metadata["B2_FLA_endpoint_contractions"] = timed(
                "endpoint_contraction_diagnostic",
                lambda: {
                    "actual_output_using_fixed_native_do": endpoint_output,
                    "current": coefficient_effect(current),
                    "reverse_in_original_orientation": coefficient_effect(reverse),
                    "average": coefficient_effect(average),
                },
            )

            def cached_average(actual_e, actual_do, actual_scale):
                metadata["additional_work"]["cached_average_callback_calls"] += 1
                require(actual_e is e, "Repropagation changed original endpoint dictionary")
                require(float(actual_scale) == scale, "Repropagation changed actual scale")
                stats = _difference_stats(actual_do, fixed_do)
                metadata["unchanged_coefficient_checks"]["cached_callback_mo_native"] = stats
                require(stats["passed"], "Repropagation changed FLA output coefficient")
                return average

            metadata["additional_work"]["GDN_repropagations_entered"] += 1
            new_gdn, new_terms = timed(
                "unchanged_GDN_surroundings_repropagation",
                lambda: gdn_finite_pullback(
                    layer.linear_attn, c, e, terms["m_mixer_output"],
                    scale, cached_average, diagnostics=True,
                ),
            )
            metadata["additional_work"]["GDN_repropagations_returned"] += 1
            # These counts follow the successfully executed unchanged helper;
            # there is no second FLA call hidden in cached_average.
            metadata["additional_work"].update(
                native_FLA_adjoint_stage_calls=2,
                public_paired_conv_preactivation_calls=1,
                public_conv_autograd_calls=1,
                conv_count_basis="Successful unchanged gdn_finite_pullback body.",
            )
            for name in ("mnorm", "mo_before_cast", "mo_native", "mz"):
                metadata["unchanged_coefficient_checks"][name] = _difference_stats(
                    new_terms[name], mixer_terms[name]
                )
            require(all(x["passed"] for x in metadata["unchanged_coefficient_checks"].values()),
                    "A coefficient upstream of the changed FLA inputs drifted")
            require(len(fla.calls) == before_call_count + 1,
                    "Unexpected additional finite FLA invocation")
            require(metadata["additional_work"]["cached_average_callback_calls"] == 1,
                    "GDN repropagation did not consume the average exactly once")
            metadata["additional_work"]["input_norm_residual_calls"] += 1
            candidate_new = timed(
                "unchanged_input_norm_residual",
                lambda: boundaries.norm_residual(
                    d["input_norm_input"][0::2], d["input_norm_input"][1::2],
                    layer.input_layernorm.weight, new_gdn,
                    terms["m_mixer_output"], layer.input_layernorm.eps,
                ),
            )
            require(bool(torch.isfinite(candidate_new).all()), "Nonfinite candidate input coefficients")
            metadata["B2_decoder_endpoint_contractions"] = {
                "fixed_output": paired_contraction(upstream, d["output"]),
                "current_input": paired_contraction(current_new, d["input_norm_input"]),
                "candidate_input": paired_contraction(candidate_new, d["input_norm_input"]),
            }

            def export():
                delta = d["input_norm_input"][1::2].double() - d["input_norm_input"][0::2].double()
                signed = (candidate_new.double() * delta).sum(-1).detach().cpu()
                reverse_cpu = {name: reverse[name].detach().to("cpu", copy=True)
                               for name in PROPAGATED_KEYS}
                return signed, reverse_cpu

            candidate_signed, reverse_cpu = timed("CPU_result_and_reverse_coefficients", export)
            metadata["CPU_artifact_bytes"] = {
                "candidate_signed": candidate_signed.numel() * candidate_signed.element_size(),
                "reverse_coefficients": sum(x.numel() * x.element_size() for x in reverse_cpu.values()),
            }
            metadata["reverse_call"] = dict(fla.calls[-1])
            metadata["status"] = "layer0_order_average_computed"
            return candidate_signed, reverse_cpu, metadata
    except EndpointOrderDiagnosticError:
        raise
    except Exception as error:
        metadata["status"] = "failed_execution"
        metadata["execution_error"] = type(error).__name__ + ": " + str(error)
        raise EndpointOrderDiagnosticError(
            "Endpoint-order diagnostic execution failed", metadata
        ) from error
    finally:
        torch.cuda.synchronize(device)
        metadata["complete_diagnostic_seconds"] = time.perf_counter() - started
        metadata["additional_finite_FLA_call_records"] = [
            dict(x) for x in fla.calls[before_call_count:]
        ]

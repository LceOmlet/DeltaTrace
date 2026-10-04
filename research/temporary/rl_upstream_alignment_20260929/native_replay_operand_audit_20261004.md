# Qwen3.5 root/replay operand audit, 2026-10-04

Read-only source and saved-receipt audit. No production or candidate numerical
implementation changed; no remote, model, GPU or dependency operation ran.
This records the existing execution interface, not a replacement RL plan.

## Provenance

The verified runtime is `c9cd147`, reference `fc2e6c2`; its runner SHA is
`c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1`.
This was independently matched to `git show c9cd147` and
`experiments/rl/verified_runtime.json` (`bcd882c704ce96997f56091cf120c9560c480a4c5bfa44fad2576dc29b9c2346`).

The saved hot B4 diagnosis is `a6b042a`, not a production deployment.
Its actual staged runner candidate is
`qwen35_dense_finite_runner_candidate.py`, SHA
`6348ebef115ff0ea45deeee3df8b65dcfbf0d1ae86619edc67f34d16d92d69aa`;
the current local runner file has those same bytes. Its new prefix-provider seam
does not add root intermediate captures. References below show
`c9cd147 line / current candidate line` when their line numbers differ.

Unchanged numerical/capture sources, matched to the verified manifest and Git:

| Local file under `deltatrace/clean/qwen35` | SHA256 |
| --- | --- |
| `qwen35_decoder_finite.py` | `047c38e6b180adb350083ff370693ed20b95e2df82e4a78cd59038bc0803a197` |
| `qwen35_gdn_finite.py` | `fcbfd9e74a6c40c3bf970d7d24ec42e0f08e5d513ff49eb1ade6a5fcd52e9fca` |
| `native_attention_capture.py` | `2a226ac35302e933920aa096fe962db7ed724c1a478c247ef5700b253461ab74` |
| `native_dense_attention_capture.py` | `d02c51325e852289358b74f5756f285af0148915c9a3a0c2d8c535a945945604` |
| `finite_fla_gpu.py` | `f1555736d32974668676eff4e040a500f4c2b907e6ac1b298f01dc2ba87f7db6` |

The hot diagnosis actually imported `deltatrace_rollout.py` SHA
`b603a5a062452c739e3cb03cd1c565347e89f0503f7ac81cf11808b102d0a034`.
The verified integration file has SHA
`d9c20b78da5e41b6e7113bd4e22b4db464fb00ed314d7a957d4455137145f9f5`.
AST comparison against `c9cd147` confirms the three FSDP lifecycle methods are
unchanged: `replay_finite_layer`, `prepare_finite_layer`,
`release_finite_layer`. Their AST SHA256 values respectively are
`a9a04f0151a35ab940bc0011593532de85ac29a102b689ba295503aedaa3d886`,
`1a29581e59d5bff32ef53dcf2cc5455fb97e21f3e0d6406abbfd0bbded906f54`,
`d9bd60f565c2950324e412e1d3582863dbfe5ccaa30392c1fb5c1bde5c96956e`.

## What the current root already supplies

Runner `218-230 / 234-245` captures each decoder's incoming hidden state and
kwargs, final norm input, and packed head input. It does **not** invoke the
decoder, FA or GDN capture classes in that root pass.

`root[str(i)]`, restored through the runner's existing CPU-to-device copy,
can supply the exact `input_norm_input` value to the existing decoder finite
rule. The next root checkpoint supplies the decoder output for
diagnostics. The saved kwargs already supply original position embeddings,
mask and positions. Final norm/head/seed handling is already separate from
decoder replay and stays unchanged.

The next checkpoint cannot directly supply `post_norm_input`, MLP projections
or mixer operands. Subtracting a later residual endpoint to reconstruct them
would introduce another finite-precision computation and is not an existing
capture interface.

## Required artifacts that root can capture through existing DT owners

| Consumer | Required existing artifact | Current owner interface / consuming lines |
| --- | --- | --- |
| Decoder residual/norm | `input_norm_input`, `post_norm_input` | `NativeDecoderCapture`, decoder file `13-43`; finite consumer `236-243` |
| MLP finite rule | `gate_output`, `up_output`, `silu_output` | Same capture; consumer `228-230` |
| FA finite rule | `q_proj_output`, native pre-gate `attention_output`, `query`, full historical `key`, full historical `value`, `q_norm_input`, `k_norm_input` | `NativeDenseAttentionCapture` / base `NativeAttentionCapture`; decoder consumer `192-216` |
| Public FA LSE call | `dense_q/k/v`, `dense_arguments`; `dense_aliases` preserves original transpose aliases | Dense capture `25-55`; runner `321-327 / 337-343`, deferred-offload path `347-354 / 363-370` |
| GDN input/scalars/conv | `projected_qkv`, `conv_output`, `raw_q`, `raw_k`, `a`, `b`, `z`, `mask` | `NativeGDNCapture` `38-130`; GDN consumer `214-290` |
| GDN native FLA endpoint bank | `q/k/v`, `raw_g`, cumulative `g`, `beta`, `A`, `w`, `v_new`, incoming chunk `h`, native norm input `o`; original scale, input shape and coefficient start | GDN capture `96-115`; GDN consumer `214-263`; `finite_fla_gpu.py` `65-87`, `136-142` |

For propagation without a full-trace observer, decoder norm outputs,
`down_input`, MLP/decoder outputs are not needed by the finite rule. GDN
`norm_output` and `output` are explicitly diagnostic-only (`116-120`). These
are already omitted by the existing retained-name/capture-output options;
there is no need for a new operator or inference path to omit them.

The existing finite consumers receive `dc.values`, `mc.values`, `mc.endpoints`
and metadata. They do not require the native layer to have been run immediately
before them. Capturing those exact original artifacts during root would avoid
the second native input norm, mixer projections/conv/FA/FLA/norm/gating/out
projection, post-attention norm and MLP gate/up/SiLU/down forward. The finite
rules and their parameter access still run. Merely retaining current root
checkpoints is insufficient to achieve that skip.

Root-side capture must use one existing capture scope at a time per executing
layer. Entering all mixer capture contexts together is not supported:
`NativeAttentionCapture`/`NativeGDNCapture` reject an existing Python profiler,
and the accelerated code-local backend similarly owns one scoped observation.
`qwen35_code_local_capture.py` SHA
`993ae72a49dc90901bfcbb7817e5fd32ad6004fbe630a190b775c6f1e77fbe72`
already composes the original retained classes and `LocalCaptureEvents`.
Original capture/close ordering and exception cleanup must be retained rather
than replacing model/FA/FLA calls. No such root-tape controller is implemented
or deployed by this audit.

Transport is not already wired for a persistent whole-root decoder tape:
the current runner constructs each decoder capture on GPU, and its finite
consumer expects device operands. A CPU-root tape would need to restore the
same captured fields at that layer's finite boundary using existing copy
helpers. `NativeDecoderCapture.retain` currently uses ordinary `.to(...)`;
unlike mixer capture it exposes no pinned-host option. This is a controller
lifetime/transport gap, not a missing finite formula or a reason to copy an
upstream forward implementation.

## Replay side effects and work that remains

* Current replay uses a prefix Cache separate from root's native fork. Its
  final K/V/conv/recurrent states are not read by any finite consumer after
  capture. The finite rules consume the captured full K/V and native chunk
  `h` directly. Eliminating the second layer forward therefore eliminates
  its additional Cache updates, while root's original native Cache transitions
  remain. This does not justify manufacturing a different cache/state.
* FA root capture does not contain LSE. The current public
  `flash_attn_func(..., return_attn_probs=True)` call remains necessary for the
  unchanged finite interface. Its native q/k/v, settings and original FA
  assertions remain. No claim is made that current captures expose another
  public zero-extra-call LSE interface.
* GDN's paired public `causal_conv1d_fn(..., activation=None)` and
  `torch.autograd.grad` at `272-279` are part of **finite propagation**, not
  native replay. Saved fused SiLU endpoints do not supply that linear
  preactivation/autograd computation. Its discarded weight-gradient work is
  also still present.
* Norm secants, finite MLP maps, finite FA/FLA recurrence and input projection
  transposes remain. FA's full historical K/V coefficient work remains even
  when decoder root/replay works only on the changed suffix.
* All 32 FSDP `prepare_finite_layer` / `release_finite_layer` operations remain.
  Without the native replay's FSDP forward hook, explicit `unshard()` at
  prepare becomes the weight gather instead of an already-satisfied call.
  Communication/HtoD is not eliminated by deleting the replay label. The
  existing forward-prefetch candidate is triggered by native forward hooks;
  its measured behavior cannot be claimed unchanged when those calls vanish.
* Current `replay_relative_L2`/`replay_output_effect` compare a real second
  output against root. A skipped replay has no second output; those fields
  cannot be populated with fabricated equality/zero. Root endpoints and
  finite diagnostics themselves remain available.

## Measured scale and storage tradeoff

The saved hot profile uses original rows 40-43 and the full 88-row capture bank,
LoRA 8/16, DT/actor B4 per rank, max length 32768. With shared prefix, its
native root takes 0.9107/0.9100 s; replay 1.1528/0.9420 s; finite decoder
1.2958/1.4816 s; public FA LSE 0.04608/0.05003 s. Profiler instrumented wall
and bank-preparation totals are not steady DT timing. Those replay phase
numbers include FSDP/capture work and are **not** an achievable saving.

The associated device-event decomposition records replay MM device durations
of 0.280191/0.304387 s. This is a measured repeated-compute subset, not a
wall-time speedup. Separate streams and communication overlap prohibit adding
all device-duration totals as wall time.

The subsequent original input observation found all 96 MLP base projection
inputs per rank equal between root and replay, including dtype/stride/device
and gathered base-weight metadata, for this same representative B4. It is
evidence of repeated work in this sample, not a universal numerical guarantee.
Its paired suffix widths are 576/610, hidden width 4096, intermediate width
12288, and native BF16 storage uses two bytes per element.

Five retained decoder fields have logical payload
`32 * 8 * S * (2*4096 + 3*12288) * 2` bytes: 12.375/13.10546875 GiB.
Sharing the already-kept root input leaves four additional fields,
`32 * 8 * S * (4096 + 3*12288) * 2`: 11.25/11.9140625 GiB.
These sums exclude FA/GDN artifacts, Cache, weights, finite working tensors
and allocator overhead. They are not measured concurrent physical peaks.

The earlier **different** actual c9cd147 storage observation has suffixes
744/749 and original full lengths 8232/8237. Summing each capture's observed
unique storage metadata over all layers gives decoder 15.984375/16.091796875
GiB, FA 3.5537109375/3.5653076171875 GiB and GDN
14.126495361328125/17.133487701416016 GiB. These are sequential capture
payload totals; they do not prove all retained tensors can coexist, nor imply
identical bytes for the present hot B4. The GDN record includes actual FP16
native FLA operands and BF16 norm input; dtype must not be inferred from the
model's parameter dtype alone.

Keeping an entire 32768-position decoder tape would already be 704 GiB for
the five BF16 fields alone (640 GiB additional when sharing layer inputs).
Current prefix reuse normally makes the suffix much shorter; even that cannot
justify an unmeasured all-GPU tape. Existing capture transport can place
artifacts on CPU, but its DtoH/HtoD and storage-lifetime cost must be measured
against the native work it removes. This audit establishes the owner seam and
operand sufficiency only, not capacity or speed acceptance.

## Receipt identities

* `results_native_prefix_hot_profile_20261004.json`: SHA
  `30cdd5edeea065684ca6bcf9cdb55f718cb2c0edccc0b042c3e36dc6551dbbca`.
  Original trace SHA rank0
  `e330c5f11d856c1c5f08d135f8b6b9f7a7e054bd2e3cbcc9cb66e136231f9997`,
  rank1 `199c185810e0b357b8c29f06c26176a6161b6b4b4cf8b80e639615b6d651cd8c`.
* `results_native_projection_inputs_20261004.json`: SHA
  `f8208debfc0980b1eb0201b1eebd3f86a0a476ae7d8fe7f368f2954592b3ea68`.
* `results_native_hot_device_breakdown_20261004.json`: SHA
  `5990071badf188a7c6415f2d8acd04bb9637636186d8fe04ef1bff862ccd1f1d`.
* `phase-observation-20261002/formal-dt-capture-storage-1790953114/captured-1790956298.json`:
  SHA `326e92dbafa31d75a7626ffabac6ecb99625c48684883c90520ce6f52335b4cd`.

No new thresholds, correction, reference calculation or numerical acceptance
claim is introduced. Existing FA/FLA references and assertions remain scoped
to their original operators. No historical Qwen3 result is evidence that the
Qwen3.5 root-tape path works.

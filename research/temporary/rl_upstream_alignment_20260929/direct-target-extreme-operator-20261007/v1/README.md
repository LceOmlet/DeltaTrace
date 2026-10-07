# Extreme AppWorld newline: original operator diagnosis

This continues the actual-token investigation using the original real AppWorld
B4, source58209daa/nativeaaa03be7, fresh original VERL actor LoRA8/16 and B4 per
rank. The unchanged two original DT calls finish; no observer-mode path, alternate
finite rule, model forward, rollout, optimizer update, checkpoint or formal restart
is added. PLAN/Q/V/A/PPO/whitening remain unchanged. The author cumulative deletion
and RISE/MAS evidence is complementary and is not replaced by this diagnosis.

Driver200641/birth1791385210.51, physical4/5, workers204712/206232. All four whole
signed vectors (single/joint on both ranks) exactly match the previous original
calls, maxabs0. Passive hooks and wrappers return original objects unchanged.
Source files, actual imported owners, effective config and SHA transport are saved.
Large original tensors remain on the remote host with SHA256; they are not copied
to the local workstation or Git. No new model inference/DT is used for analysis.

## Measured output-end difference

The native cached single-EOS target effect is+20.71933239. The original joint
log-prob seed contracted with this same actual single-deletion logit difference
is+7.98411942 using CPU FP32 and+7.98411319 using CPU FP64. The dominant immediate
code-fence target alone changes+21.33316231 natively, but the joint seed assigns
+6.65016737 along this single-deletion direction. This discrepancy persists with
FP64 evaluation of the unchanged original seed; increased analysis precision does
not remove it. The joint finite endpoint identity itself is accurate in FP64:
100.45795617519036 versus100.4579561751856. Neither identity nor aggregate
conservation makes the joint coefficient an exact single-deletion coefficient.

After the actual original BF16 head transpose the single-direction contraction
is+8.51523615; after final norm it is+7.61414271. Native factual logits and
normalized hidden endpoints are exactly equal between both calls. Projection,
norm and reference-context effects are kept separate; no multiplier corrects them.

## Decoder27 split

Both ranks agree exactly. Output contraction+.99628511; original MLP coefficient
on single post-norm-output delta-.50737119; post-norm/residual input+.86740143;
after attention+.51147562; input-norm/residual input-1.58704043. At the last merge,
the residual branch is-2.22394793. Thus a first crossing of the full decoder is
not evidence that the FA kernel created a negative signal.

Original norm FP32 eager versus actual compiled coefficients differ at most
7.4505806e-9. The actual single-direction input-norm contraction remains
-1.58704043 in FP64. Original DT finite norm versus Torch functional.rms_norm
continuous FP64 endpoint contractions differ by1.42e-14 on both saved norms.
These are descriptive operator checks, not new acceptance tolerances or proof
that the joint reference estimates individual causal deletions accurately.

An earlier optional CPU HF-norm import initialized the device runtime; its
no-CUDA-init guard failed and no accepted norm result was emitted. Its script and
stderr are retained. The completed analysis instead uses the saved tensors,
unchanged original DT function and Torch functional.rms_norm; CUDA is uninitialized.
No substitute HF model or manually copied norm forward is used in that result.

## Exact existing FA assertions

The original saved-operand checker SHA7ff11d9d, original FA v2.6.3 reference source
SHAa290e11c, and unmodified AST assertions own the test. Actual row0 has full causal
q length6815 and k/v length9567, BF16 q/k/v, FP32 saved upstream, original
coefficient_start2762. Uniform original per-row query_start2752 is represented
through the existing scalar checker interface; no causal rows are shortened.
The captured native v1 was unavailable, so this check explicitly uses the actual
reference endpoint q0/k0/v0, with the actual joint upstream; it is not called a
factual-endpoint check. No values are synthesized, approximated or resampled.

Original native output, native dq/dk/dv, and coincident-endpoint finite dq/dk/dv
all pass the seven original assertions: output2x reordered baseline, gradients3x.
This does not test a nonzero finite secant against a single-deletion causal truth,
or replace the old failures on other actual BF16 dk operands. It does not prove
the entire DT estimate accurate. The attention test uses original operator
reference/autograd; it does not perform a model training backward or update.

## Training and memory status

TextCraft's first update stays held; AppWorld stays terminal. This investigation
has not repaired the evidenced joint-reference/single-deletion mismatch or
released either task for training. No clipping, sign correction or reward change
is used. The separate storage repair79922486 remains bounded-verified on the
original failing B4 with original async vLLM: physical55.716/55.290GiB, longest
context27334, full signed/QVA exactly equal to the previous offload comparison.
It is not formally deployed, exact32768 or full28-batch verification.

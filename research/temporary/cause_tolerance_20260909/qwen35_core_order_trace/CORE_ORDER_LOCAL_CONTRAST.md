# Falsify the baseline-content control explanation before changing all DT

The matched native pilot puts most mixer error inside finite FA/FLA cores;
Q/K normalization, output gates and projection rounding are smaller. A concrete
remaining explanation is the core's content1 order: control changes contract
against baseline content/state while content follows original-input dynamics.
Actual partial deletion retains different content. This is an allocation choice,
not a demonstrated model/kernel error or proof that reversing it wins.

Test one uniform alternative at every Qwen3.5 core, with the current upstream
and all actual native states fixed. Reverse the two endpoints in the SAME
finite core. FA then uses `deltaP*V1 + P0*deltaV`. FLA uses the opposite ordered
finite recurrence, including its actual endpoint0 state transition and endpoint1
content contractions. No coefficient sign reversal is needed: reversing both
finite differences preserves the transpose map's orientation to original deltas.
Outer gates/norms/MLP and actual model execution do not change. Never interpret
mixed factors as real counterfactual model forwards.

This local alternative has the same normal core operation count and memory
order, plus linear endpoint layout work. Its actual whole-call cost and quality
are unknown. The diagnostic pays one ADDITIONAL core call per layer and is not
a timing comparison or already integrated candidate. The Qwen3 P0 alternative
has older negative16-case quality evidence, so do not repeat that whole trial.
Qwen3.5's hybrid recurrence requires its own local check.

Use the already frozen NI0/1 MH0/1 DT/FTK1 10% deletion sets. Record each layer's
actual core-output effect, current and reversed predictions, and errors for
both sets. Do not select layers. The original complete DT vector must equal the
same-process plain control; all alternative coefficients must be finite and
come from the existing source-pinned kernels. Preserve prior-process drift.

A whole-method trial is justified by this proposed local mechanism only if
both NI cases reduce (a) excess preference for DT's set and (b) mean absolute
core error across all32layers and both sets, while MH mean absolute error does
not worsen. Otherwise stop this explanation/alternative without a metric sweep.
This gate does not prove a rejected local rule could never help a full model;
it prevents expanding a mechanism that the measured local contrast contradicts.
Even a passed gate still needs original signed RISE, positive MAS/needle,
all16cases and other author-task validation before adoption.

Cost:8completeDT+8pairedpartialnativeforwards+1nativeinitialization+1model load;
32extra finiteFA calls and96extra finiteFLA calls in the four observed passes.
Existing diagnostic norm/projection work is also charged. No new source model,
FA/FLA implementation, generation, FT attribution or21-point metric run.

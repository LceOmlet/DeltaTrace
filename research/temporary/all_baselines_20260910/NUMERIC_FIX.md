# Technical amendment before baseline quality scoring

The native first AttnLRP pilot (VT H2-C3, index 0) failed with a nonfinite
aggregate. Autograd anomaly detection located IdentityRuleImplicitFnBackward.
Inspection of its saved ratios found 11 nonfinite ratios, all at exactly zero
inputs, in 36 calls. The author's expression is output/(input+1e-10); in FP16
the epsilon rounds to zero and produces 0/0. Preserve this failure and log.

Apply one explicit runtime numeric repair to AttnLRP: retain the original
forward output and every finite saved ratio bit for bit. Only when the saved
ratio is nonfinite AND its input and output are both exactly zero, reevaluate
that ratio in FP32 using the existing 1e-10 epsilon, yielding zero. Reject all
other nonfinite cases. Restore the original function in a finally block.
Record the number of repaired ratios per case. No model weights, activations,
target weights, objective scaling, retrieval policy or hyperparameter changes.

This is an amended AttnLRP implementation and must be marked in the report;
do not describe it as wholly unchanged author code. The other four baseline
algorithms are unaffected. The new run uses a separate output directory and
retains the initial three successful technical pilot results for bitwise
control comparisons. DT/FT are not rerun. Freeze this amendment and code in Git
before inspecting any remaining baseline quality scores.

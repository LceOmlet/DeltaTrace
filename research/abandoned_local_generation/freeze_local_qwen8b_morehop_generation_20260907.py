"""Freeze local generation, with explicit differences from the author API sampler."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
A = Path(__file__).resolve().parent
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
study = A/'local_qwen8b_morehop_generation_20260907.py'
ast.parse(study.read_text())
p = json.loads((A/'original_morehop_sampler_protocol_20260907.json').read_text())
base = json.loads((A/'original_FT_seq_cost16_protocol_20260907.json').read_text())
for key in ['checkpoint', 'checkpoint_receipt', 'checkpoint_receipt_sha256', 'native_model_source_sha256']:
    p[key] = base[key]
p.update(study_sha256=sha(study), status='frozen_before_local_generation',
    purpose='User-directed local Qwen3-8B generation on unused original MoreHopQA. Same original data/prompt/format/span helpers, explicitly changed generator and blind reference-answer review. Original FT metrics will be used only after answer review and cache freeze; this is not exact original API sampling or Table1 reproduction.',
    sampling_source='${ARTIFACT_ROOT}/codex_morehop_original_confirmation_20260907_v1/sampling_original_source.json',
    executed_source_offsets=list(range(64)), batch_size=4, max_new_tokens=8192, seed=42,
    generator_model='native_local_Qwen3-8B_same_checkpoint_as_attribution', judge_model=None,
    generation_policy='Actual native model.generate, default FA, FP16, greedy, one beam, original 8192-token upper limit. Original build_gen_messages. Native tokenizer template enable_thinking=False lets the original system prompt request visible reasoning; no special-think extraction or rewriting. Explicit left padding and masks; full original prompts and outputs retained. Official native generation KV cache only, no cross-example or cross-call output reuse.',
    sampling_rules='First64 of the already frozen256 unexposed original rows, in original order, physical B4. Generate all64 regardless of correctness. No retries on semantic/format failure, no score-dependent filtering. If fewer than64 correct format/span-valid answers are obtained, later expansion may use only the next preselected raw rows, with a new execution receipt; the existing 256-row ceiling and first64-accepted target remain.',
    answer_review_policy='Before any attribution or quality evaluation, review all parsed boxed answers for semantic equivalence to provided original reference answers, recording the reason and reviewer provenance. Original judge criterion is answer equivalence, not certification of intermediate reasoning. No Qwen8B self-judge, no fabricated DeepSeek responses. Exact or conservative normalized equality can be confirmed mechanically; sentence answers need documented review, never mere substring acceptance. Uncertain/incorrect/formatted-failure/truncated outputs remain recorded. Only first64 independently of attribution accepted, format/span-valid, EOS-finished rows enter the correct-answer confirmation cache. No gold answers sent to generator.',
    source_independence_scope='Source overlap exclusion was frozen before this run and excludes previous95 contexts/base questions/IDs. It does not imply all64 are distinct question groups; report grouping. No new-cohort P1 or FT attribution is run by this script.',
    api_dependency='None. Earlier API preflight retained as historical evidence, superseded for this local-generation branch.',
    cost_policy='Report actual B4 generation calls, every native root/layer forward and physical trajectory including completed lanes still computed, wall time, output lengths and peak memory. Generation is dataset preparation, never hidden in attribution efficiency or counted as an attribution gain.')
for key in ['maximum_API_attempts','API_key_policy','cache_namespace']:
    p.pop(key, None)
protocol = A/'local_qwen8b_morehop_generation_protocol_20260907.json'
protocol.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding='utf-8')
subprocess.run([sys.executable, str(A/'prepare_remote_experiment.py'),
    '${ARTIFACT_ROOT}/codex_local_qwen8b_morehop_generation_20260907_v1',
    f'study.py={study}', f'protocol.json={protocol}', '--request',
    str(A/'launch_local_qwen8b_morehop_generation_20260907.json')], check=True)
print(json.dumps({'study_sha256': sha(study), 'protocol_sha256': sha(protocol),
    'raw_examples': 64, 'batch_size': 4, 'API_calls': 0}))

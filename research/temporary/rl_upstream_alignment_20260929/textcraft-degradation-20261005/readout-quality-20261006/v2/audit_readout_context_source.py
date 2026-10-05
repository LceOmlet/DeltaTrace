"""Bounded stdlib review of saved readout inputs and frozen owner source contracts."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
DEG = HERE.parents[1]
OLD = '7831caf8f019eec1a0f069cb5a05b1ccddc60e0b'
PRODUCER = '99fb5c28d9be064bab22dc8ad949494b3fd85150'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def artifact(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': sha(data), 'bytes': len(data)}

def frozen(commit, relative, expected, ranges):
    data = subprocess.check_output(['git', 'show', commit + ':' + relative])
    assert sha(data) == expected
    return {'commit': commit, 'repository_path': relative, 'sha256': sha(data), 'bytes': len(data), 'line_ranges': ranges}

pack = json.loads((HERE / 'original-first-response-cases.json').read_text())
decode = json.loads((HERE / 'native-prefix-decoding.json').read_text())
mapping = json.loads((HERE / 'original-readout-mapped-requests.json').read_text())
metadata = json.loads((HERE / 'original-native-collect-metadata.json').read_text())
groups = json.loads((HERE / 'independent-readout-groups.json').read_text())
cases = [case for rank in pack['rank_cases'] for case in rank]
first = {case['traj_uid']: case for case in cases}
assert len(first) == len(cases) == 64
unique = {}
duplicate_equal = []
for request in mapping['requests']:
    key = (request['traj_uid'], request['env_step'])
    if key in unique:
        original = unique[key]
        fields = ('selected_input_ids', 'reference_input_ids', 'source_start', 'source_end', 'target_id')
        duplicate_equal.append({'traj_uid':key[0], 'env_step':key[1], 'input_ID_span_target_fields_exact':all(original[field] == request[field] for field in fields), 'factual_probability_delta_second_minus_first':request['factual_probability']-original['factual_probability'], 'EOS_probability_delta_second_minus_first':request['whole_response_EOS_probability']-original['whole_response_EOS_probability']})
    else:
        unique[key] = request
requests = list(unique.values())
prefix_mismatches = []
first_response_mismatches = []
for request in requests:
    original = first[request['traj_uid']]
    initial_ids = original['selected_input_ids'][:original['source_start']]
    if request['selected_input_ids'][:len(initial_ids)] != initial_ids:
        prefix_mismatches.append([request['traj_uid'], request['env_step']])
    if request['env_step'] == 0 and request['selected_input_ids'] != original['selected_input_ids']:
        first_response_mismatches.append(request['traj_uid'])

def boundary_summary(rows):
    ends, heads, tails, thinks = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        ids, start, end = row['selected_input_ids'], row['source_start'], row['source_end']
        source, query = ids[start:end], ids[end:-1]
        ends[str(source[-1])] += 1
        heads[','.join(map(str, query[:6]))] += 1
        tails[','.join(map(str, query[-9:]))] += 1
        thinks[str((source.count(248068), source.count(248069)))] += 1
    return {'rows': len(rows), 'source_final_token_ID': dict(ends), 'query_first_six_IDs': dict(heads), 'query_last_nine_IDs': dict(tails), 'current_response_think_open_close_counts': dict(thinks)}

def probability_description(rows):
    a = [float(row['factual_probability']) for row in rows]
    b = [float(row['whole_response_EOS_probability']) for row in rows]
    delta = [x-y for x,y in zip(a,b)]
    return {'rows':len(rows), 'factual_p_success_mean':statistics.fmean(a), 'full_response_EOS_p_success_mean':statistics.fmean(b), 'factual_minus_EOS_mean':statistics.fmean(delta), 'factual_above_EOS':sum(x>0 for x in delta), 'factual_below_EOS':sum(x<0 for x in delta), 'factual_range':[min(a),max(a)], 'EOS_range':[min(b),max(b)]}

by_step = defaultdict(list)
for row in requests:
    by_step[row['env_step']].append(row)

files = ['original-first-response-cases.json', 'native-prefix-decoding.json', 'native-prefix-decoding-cpu-run.json', 'native-prefix-decoding-local-fetch.json', 'original-readout-mapped-requests.json', 'original-native-collect-metadata.json', 'original-tokenizer-config.json', 'original-tokenizer-chat-template.jinja', 'actual-agentgym-schemas.py', 'old-stopped-agentgym-schema-before-artifacts.py', 'actual-agentgym-vllm-rollout.py', 'actual-agentgym-rl-dataset.py', 'readout-context-owner-fetch.json', 'readout-history-contract-fetch.json', 'independent-readout-groups.json', 'decode_saved_readout_prefixes.py', 'run_saved_readout_prefix_decoding.py', 'fetch_readout_context_owners.py', 'fetch_readout_history_contract.py', 'audit_readout_context_source.py']
receipt = {
    'scope':'Read-only saved checkpoint25 native input/decode/source review. No new forecasts, model initialization, forward, backward, optimizer, or owner modification. This review does not identify a formatting cause or a production fix.',
    'sources':{name:artifact(HERE/name) for name in files},
    'frozen_owners':{
        'reward_readout':frozen(OLD,'experiments/rl/reward_readout.py','228afbc7a10841d482c3d73def59dfe9ef192c057a97d76dca57369502a10137',[[61,90],[92,146],[183,227],[250,280]]),
        'deltatrace_rollout':frozen(PRODUCER,'experiments/rl/deltatrace_rollout.py','0ad37a17aede30089fd2ac9a609a42689e6a3a68d00b601520db0a5cff8b8e6e',[[28,72],[231,296],[381,403],[424,448]]),
        'old_textcraft_owner_rollout':frozen(OLD,'experiments/rl/textcraft_owner_rollout.py','92c724a520aa10412e2ced92e4667cf0638be0e2a909d477e52aca75ee805e2c',[[22,44],[67,99],[109,129],[154,177]]),
        'actual_c9_dense_runner':dict(artifact(DEG/'native-layout-20261006/v1/owner-contract/qwen35_dense_finite_runner.py'), line_ranges=[[148,162]]),
        'actual_c9_answer_runner':dict(artifact(DEG/'native-layout-20261006/v1/owner-contract/qwen35_answer_finite.py'), line_ranges=[[16,41],[100,110]]),
    },
    'first_response_actual_inputs':{
        'coverage':decode['coverage'],
        'goals_and_recipe_text':'All 8 original prompt ID arrays decode to the native task instructions, supplied crafting commands, and an explicit Goal: craft ... . Goal is in the preceding prompt, not repeated in the forecast query.',
        'decoded_goals':['warped sign','spruce pressure plate','packed ice','orange concrete powder','wooden pickaxe','cauldron','granite slab','melon'],
        'initial_observation_scope':'These native initial prompts contain the task/recipes/goal; they do not contain a separately explicit initial inventory list. This is the saved original environment input, not an inferred missing-state defect.',
        'labels':{'n':2,'IDs':[15,16],'decodes':['0','1'],'semantics':['cumulative official return 0: goal not achieved before episode ends','cumulative official return 1: official environment reports goal achieved']},
        'query_policy_budget':{'sampling':{'max_tokens':512,'temperature':1.0},'max_interactions':30,'current_interaction':1,'discount':1,'context_cap':32768,'query_tokens':211},
        'target_and_causality':'The query IDs are identical across all 64 cases despite different realized returns. The observed return chooses the appended target ID after the forecast predictor. Actual read_outcomes inputs omit that target; the original dense runner selects only the two label head rows and applies their log softmax. No saved future observation, action, actual stopping time, or realized reward is inserted into these query IDs.',
        'future_scope':'Source hook snapshots prompt/current response before the current official env.step and its following state are appended (original vllm_rollout 234-245; owner hook 91-99). This source contract and these 64 saved IDs do not establish every historical runtime input.',
        'boundary':boundary_summary(cases),
    },
    'query_chat_boundary':{
        'construction':'Frozen RewardAlphabet.query_ids 132-146 directly encodes explicit Qwen message control tokens with add_special_tokens=False, then _prepare_episode appends those original query IDs. It does not invoke apply_chat_template a second time.',
        'actual_decoded_start':'<|im_end|>\n<|im_start|>user\nFuture cumulative return forecast for TextCraft.',
        'actual_decoded_end':'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
        'predictor_state':'The label predictor follows the new assistant message and a closed empty think block. The prior current response belongs to the preceding assistant role; 38/64 contain one think opener and no think closer, 26/64 contain both. Its stored final im_end is followed by the query initial im_end in all 64, and in all 186 formal requests.',
        'contract_boundary':'Explicit role delimiters are part of the frozen readout owner contract. The saved double im_end and unfinished prior reasoning are measured formatting facts; there is no corresponding owner assertion or matched alternate-score evidence here proving them defective or causal. No template or token changes are proposed.',
    },
    'formal_existing_requests':{
        'transport_slots':len(mapping['requests']), 'unique_traj_uid_env_step':len(requests), 'successful_trajectories':len(set(x['traj_uid'] for x in requests)), 'padding_duplicates':len(duplicate_equal), 'duplicates':duplicate_equal,
        'deduplication':'Descriptive unique-request statistics retain the first saved occurrence of each (traj_uid, env_step). All six duplicate input ID/span/target tuples are exact; their small saved probability differences are reported without a tolerance verdict or a claim of an identical numerical execution.',
        'all_G':sorted(set(float(x['trace']['observed_return']) for x in requests)),
        'initial_native_prompt_ID_prefix_mismatches':prefix_mismatches,
        'matched_success_first_response_full_input_mismatches':first_response_mismatches,
        'Goal_retention':'All 186 factual inputs retain, ID for ID, their matched decoded native initial task prefix from the saved64. Their 21 first response full inputs also exactly match the corresponding saved64 inputs.',
        'context_token_range':[min(len(x['selected_input_ids']) for x in requests), max(len(x['selected_input_ids']) for x in requests)],
        'boundary':boundary_summary(requests),
        'probability_all_G1':probability_description(requests),
        'probability_by_source_step':{str(k):probability_description(v) for k,v in sorted(by_step.items())},
        'scope_limit':'All 186 formal requests are from the 21 successful trajectories; they cannot quantify failed-state calibration. These stored target log probabilities and probabilities are not a complete pair of pre-softmax label logits.',
        'native_layout_boundary':'The original formal dense paired DT requests and the later four-row read_outcomes probes share these matched factual token IDs, but use different saved native forward layouts. Their probabilities are reported separately; equal IDs do not establish an identical numerical path or assign responsibility for cross-path differences.',
    },
    'official_carrier_vs_generation':{
        'actual_schema_equals_frozen_backup':(HERE/'actual-agentgym-schemas.py').read_bytes()==(HERE/'old-stopped-agentgym-schema-before-artifacts.py').read_bytes(),
        'original_carrier':'Original AgentGym schema 83-119 encodes assistant content, preserves it in training input IDs, and appends im_end with loss mask 1; 122-161 appends the official current environment state as masked user text. The bridge reads these original IDs/masks rather than constructing a new history. Original truncate_output_ids 163-171 controls final retained training arrays.',
        'original_generation':'Original schema 76-80 renders the entire messages list through original tokenizer.apply_chat_template. Original Qwen template 94-104 removes parsed reasoning from previous assistant content when </think> occurs; 147-153 opens a think block for generation. Original dataset 113-118 puts a system message into manually rendered training IDs while raw messages contain only user/assistant; template 62-65 emits system only when explicitly present.',
        'separate_scope':'These are source-evidenced differences already in the official training carrier and generation paths; DT consumes the official training carrier. They are not evidence of a DT-only introduced loss/format bug or of the magnitude of actual sampling-prefix differences.',
        'saved_sampling_ID_gap':{'metadata_handlers':len(metadata['handlers']),'metadata_turns':sum(len(x['turns']) for x in metadata['handlers']),'saved_turn_fields':sorted({k for h in metadata['handlers'] for t in h['turns'] for k in t}),'meaning':'native_prompt_ids_length is the training prefix length. transport_output_ids_length is generated output length, not sampling prompt length. The saved metadata and request map omit EngineTransport prompt_token_ids. No matched sampling/training prefix lengths, hashes, or differences are reconstructed.'},
    },
    'quality_observation':{
        'first64':groups['all'], 'by_actual_return':groups['by_observed_return'], 'within_group':groups['group_description'],
        'bounded_interpretation':'The saved readout has weak within-task outcome separation and high success probabilities on this batch. Probabilities differ among saved tasks/contexts/steps, which establishes that outputs are not constant without isolating context from native layout/storage effects. Within the original formal G1 requests, step0 mean factual-minus-EOS is negative, whereas all186 mean is positive; first-response EOS optimism is not universal across formal requests. Fixed 0/1 labels and one template cannot identify how much comes from label vocabulary preference versus forecasting quality; Brier/calibration mismatch alone is not an interface bug.',
        'current_priority':'These formatting/source facts do not supersede the separately measured joint-to-single decomposition error and weak task-gradient evidence, and do not establish a repair requirement or historical causal proof.',
    },
    'operations':{'new_models':0,'new_model_forwards':0,'new_sampling':0,'new_backward':0,'optimizer_updates':0,'production_edits':0,'label_remapping':0,'template_candidates':0},
    'limits':['Only the original tokenizer was loaded for the prior saved-prefix CPU decode; runtime/CUDA/resource evidence is in the bound CPU run receipt. This source review and JSON reductions use stdlib only.','No sampling input was reconstructed, no environment parser was replaced, no future observations were inferred, and no numeric pass threshold was invented.'],
}
out = HERE / 'readout-context-label-source-audit.json'
out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'path':str(out),'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'formal_unique':len(requests),'initial_prefix_mismatches':len(prefix_mismatches),'first_response_mismatches':len(first_response_mismatches)},sort_keys=True))

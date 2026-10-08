"""Record frozen collection work and the user's withdrawn refinement proposal."""
from pathlib import Path
import hashlib
import json
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def binding(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(REPO).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw)}


def main():
    disposition = json.loads((HERE/'candidate-status.json').read_bytes())
    disposition['retained_research_data'] = disposition.pop('accepted_artifacts',
        disposition.get('retained_research_data', []))
    (HERE/'candidate-status.json').write_text(json.dumps(disposition, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    manifest = json.loads((HERE/'manifest.json').read_bytes())
    corpus = json.loads((HERE/'corpus.json').read_bytes())
    composed = json.loads((REPO/'experiments/rl/results_gdn_context_composition_20261008.json').read_bytes())
    jacobian = json.loads((REPO/'experiments/rl/results_native_factual_jacobian_20261008.json').read_bytes())
    verification = json.loads((HERE/'cpu-verification.json').read_bytes())
    result = {
        'status': 'Collection/split groundwork verified on CPU; output-refinement candidate rejected before GPU launch',
        'unix': time.time(), 'scope': 'Research infrastructure and existing-evidence derivation, not an attribution repair or collection-quality result',
        'tasks': {task: {'counts': data['counts'], 'source': data['source']} for task, data in manifest['tasks'].items()},
        'corpus_runtime': corpus['runtime'],
        'aggregation': manifest['aggregation'],
        'evaluation_owner': manifest['metric_owner'],
        'current_research_direction': {
            'observed': 'At the saved current GDN30 case, the original finite rule matches its own endpoint pair. '
                'The measured discrepancy is joint-pair coefficients applied to the actual single-intervention displacement. '
                'This locates an approximation-context error at that sub-operation, not an all-token error rate or an official kernel failure.',
            'measured_context_difference': composed['owner_self_comparisons']['joint_coefficients_on_single_delta'],
            'factual_gradient_is_not_a_finite_repair': jacobian['summary'],
            'exact_coupled_identities_to_preserve': {
                'state': 'H_t = M_t H_(t-1) + B_t; for GDN, M_t = alpha_t (I - beta_t k_t k_t^T), B_t = beta_t k_t v_t^T.',
                'finite_state_difference': 'Delta_i H_t = M_t^F Delta_i H_(t-1) + Delta_i M_t H_(t-1)^F - Delta_i M_t Delta_i H_(t-1) + Delta_i B_t.',
                'readout': 'Delta_i o_t = (q_t^F)^T Delta_i H_t + (Delta_i q_t)^T H_t^F - (Delta_i q_t)^T Delta_i H_t, with the original scale retained.',
                'meaning': 'Delta_i is factual minus the same single intervention, not the all-sources endpoint difference. '
                    'These are algebraic identities identifying coupled terms, not an implemented estimator or a claim of a new one-DT exact algorithm.',
            },
            'next_mathematical_question': 'Determine how the existing finite owner can approximate these coupled single-source terms '
                'within one batched reverse computation, without an all-source intervention-state bank or extra whole-model query layer.',
            'implementation_ready': False,
            'forbidden_shortcuts': ['selected extreme-output replacement', 'V-only substitution',
                                   'factual VJP presented as exact finite effect', 'forced conservation/sign/scaling'],
        },
        'withdrawn_candidate': disposition,
        'cpu_contract_checks': {'passed': verification['passed'], 'failed': verification['failed'],
                                'scope': 'Actual stored corpus split, duplicate/missing handling and owner-score aggregation only'},
        'artifacts': {name: binding(HERE/name) for name in (
            'corpus.json', 'manifest.json', 'cpu-verification.json', 'candidate-status.json',
            'collect_corpus.py', 'freeze_manifest.py', 'score_summary.py', 'verify_protocol.py')},
        'evidence': {name: binding(REPO/'experiments/rl'/name) for name in (
            'results_gdn_context_composition_20261008.json', 'results_native_factual_jacobian_20261008.json')},
        'operations': {'new_model_calls': 0, 'new_DT_calls': 0, 'new_optimizer_steps': 0,
                       'new_GPU_job_started': False, 'production_modified': False,
                       'formal_restart': False, 'checkpoint_restore': False,
                       'TextCraft_update_released': False, 'credit_repaired': False},
    }
    path = REPO/'experiments/rl/results_credit_research_collection_20261008.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    receipt = binding(path)
    snapshot = REPO/'experiments/rl/current_runtime.json'
    key = 'latest_collection_credit_research'
    original = snapshot.read_bytes()
    existing = json.loads(original)
    if key in existing:
        raise ValueError('Do not overwrite the recorded research revision')
    entry = {key: {'receipt': receipt, 'candidate_withdrawn': True,
                   'frozen_manifest_sha256': binding(HERE/'manifest.json')['sha256'],
                   'research_authorized': True, 'new_GPU_job_started': False,
                   'credit_repaired': False, 'production_modified': False}}
    suffix = '\r\n'.join(json.dumps(entry, ensure_ascii=False, indent=2).splitlines()[1:-1]).encode()
    assert original.endswith(b'}\r\n')
    snapshot.write_bytes(original[:-3].rstrip(b'\r\n')+b',\r\n'+suffix+b'\r\n}\r\n')
    print(json.dumps({'receipt': receipt, 'manifest_sha256': entry[key]['frozen_manifest_sha256'],
                      'new_GPU_job_started': False}, indent=2))


if __name__ == '__main__':
    main()

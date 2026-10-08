"""Check saved real readout inputs with each actual owner's row preparation.

This is a CPU boundary audit, not a DT, model or PPO numerical test. The
original full signed vector is read, never reconstructed or corrected. It
does not certify that a non-policy mask was assigned by the environment
correctly, or that the absent reference argument was preserved at runtime.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import psutil
import torch


SOURCES = {
    'textcraft': '2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52',
    'appworld': '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0',
}


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    started = time.perf_counter()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_bytes())
    result = dict(scope=__doc__, unix=time.time(), inputs=[ref(plan_path), ref(__file__)],
        tasks={}, new_model_queries=0, new_DT=0, new_optimizer=0, production_modified=False,
        CUDA_initialized=False, verification_scope='Saved mask/packing identity only; no numerical accuracy claim')
    process = psutil.Process()
    peak_pss = 0
    for task, spec in plan['tasks'].items():
        source_path = Path(args.root)/f'runs/direct-target-prefix-runtime-20261007-v1/{task}/{task}-dt/source.json'
        assert ref(source_path)['sha256'] == SOURCES[task]
        source = json.loads(source_path.read_bytes())
        imports = source['actual_CPU_imports']
        sys.path[:0] = source['pythonpath'].split(':')
        owner_path = Path(imports['reward_readout']['path'])
        assert ref(owner_path)['sha256'] == imports['reward_readout']['sha256']
        module_spec = importlib.util.spec_from_file_location('saved_mask_owner_'+task, owner_path)
        owner = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = owner
        module_spec.loader.exec_module(owner)
        dependency_imports = {}
        for name in ('counterfactual', 'deltatrace_credit'):
            imported = sys.modules[name]
            dependency_imports[name] = ref(inspect.getsourcefile(imported))
            if name in imports:
                assert dependency_imports[name]['sha256'] == imports[name]['sha256']
        prepare = owner.DirectActionTargetReadout._prepare_row
        grouped = {}
        for entry in spec['entries']:
            grouped.setdefault(entry['native']['path'], []).append(entry)
        rows, files = [], []
        for path, entries in grouped.items():
            actual_ref = ref(path)
            assert all(actual_ref['sha256'] == e['native']['sha256'] for e in entries)
            files.append(actual_ref)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['provenance']['source_sha256'] == SOURCES[task]
            by_uid = {str(r['traj_uid']): r for r in saved['rows']}
            for entry in entries:
                captured = by_uid[entry['traj_uid']]
                prepared = prepare(captured['row'], captured['trajectory_index'])
                fields = ('selected', 'suffix_positions', 'policy', 'target', 'prior')
                matches = {name: torch.equal(prepared[name], captured[name]) for name in fields}
                assert all(matches.values())
                assert prepared['prompt_length'] == captured['prompt_length']
                assert prepared['target_offsets'] == captured['target_offsets']
                signed = saved['native_signed'][captured['batch_row']]
                n, start = prepared['selected'].numel(), prepared['prompt_length']
                # Align saved coefficients to the exact captured packed suffix.
                # This reads their existing coordinates; no credit is recomputed.
                suffix = prepared['suffix_positions']
                valid_policy = prepared['policy'][suffix]
                valid_target = prepared['target'][suffix]
                valid_prior = prepared['prior'][suffix]
                values = signed[start:n]
                unexpected = int(torch.count_nonzero(values[~valid_prior]))
                before_prompt = int(torch.count_nonzero(signed[:start]))
                after_end = int(torch.count_nonzero(signed[n:]))
                assert not (unexpected or before_prompt or after_end)
                assert not bool((prepared['prior'] & ~prepared['policy']).any())
                assert not bool((prepared['prior'] & prepared['target']).any())
                rows.append(dict(traj_uid=entry['traj_uid'], first_stage=entry['first_stage'],
                    initial_state_sha256=entry['initial_state_sha256'], owner_fields_equal=matches,
                    source_mask_tokens=int(valid_prior.sum()), target_mask_tokens=int(valid_target.sum()),
                    nonpolicy_suffix_tokens=int((~valid_policy).sum()), packed_tokens=n,
                    original_signed_nonzero_outside_prior=unexpected,
                    original_signed_nonzero_prompt=before_prompt,
                    original_signed_nonzero_right_padding=after_end))
            peak_pss = max(peak_pss, process.memory_full_info().pss)
            del saved
        groups = {}
        for primary in (True, False):
            selected = [r for r in rows if r['first_stage'] == primary]
            groups['primary' if primary else 'extra_tail'] = dict(trajectories=len(selected),
                states=len({r['initial_state_sha256'] for r in selected}),
                source_mask_tokens=sum(r['source_mask_tokens'] for r in selected),
                target_mask_tokens=sum(r['target_mask_tokens'] for r in selected),
                nonpolicy_suffix_tokens=sum(r['nonpolicy_suffix_tokens'] for r in selected),
                all_owner_fields_equal=all(all(r['owner_fields_equal'].values()) for r in selected),
                unexpected_signed_entries=sum(r['original_signed_nonzero_outside_prior']+
                    r['original_signed_nonzero_prompt']+r['original_signed_nonzero_right_padding'] for r in selected))
        result['tasks'][task] = dict(source=ref(source_path), owner=ref(owner_path),
            actual_dependency_imports=dependency_imports, files=files,
            groups=groups, rows=rows,
            limitation='Actual reference IDs were not in these preserved captures. The pinned trajectories '
                'source replaces only prepared prior positions, and saved coefficients outside those positions '
                'are zero. This boundary result does not prove the finite approximation accurate.')
        print(json.dumps(dict(task=task, groups=groups)), flush=True)
    assert not torch.cuda.is_initialized()
    result.update(elapsed_seconds=time.perf_counter()-started, sampled_peak_process_pss_bytes=peak_pss)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=ref(args.output), elapsed_seconds=result['elapsed_seconds'],
        sampled_peak_process_pss_bytes=peak_pss)), flush=True)


if __name__ == '__main__':
    main()

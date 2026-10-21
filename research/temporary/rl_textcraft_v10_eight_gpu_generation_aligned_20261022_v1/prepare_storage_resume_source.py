"""Prepare an isolated owner initialization fix; do not launch or resume training."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OLD = ROOT / 'candidates/textcraft-v10-eight-gpu-generation-aligned-20261022-v2'
CANDIDATE = ROOT / 'candidates/textcraft-v10-eight-gpu-native-resume5-resource-20261022-v1'
MODE = 'root_pageable_replay_pinned_resident16384_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bridge = CANDIDATE / 'entry/accepted_dt_storage_profile.py'
    assert sha(bridge) == '763143c13e589612408a8712dbb3a917fe38475fe11b7041a6591afdb82bea22'
    target = CANDIDATE / 'verl-owner'
    assert not target.exists(), 'Inspect any partial source preparation before continuing'
    for child in (OLD / 'entry').iterdir():
        if child.name == '__pycache__':
            continue
        assert child.name != bridge.name
        (CANDIDATE / 'entry' / child.name).symlink_to(child.resolve())
    shutil.copytree(OLD / 'verl-owner', target, symlinks=True, copy_function=os.link,
                    ignore=shutil.ignore_patterns('.git', '__pycache__'))
    parent = OLD / 'verl-owner/verl/workers/fsdp_workers.py'
    worker = target / 'verl/workers/fsdp_workers.py'
    assert sha(parent) == '198f509c9a0af8a03d54f22e66f5f0d5593ab4c9eb7b27777c201ef323d2fecb'
    before = worker.read_text()
    anchor = '                )\n            return self._deltatrace_producer.attribute_prepared_batch(data)'
    insertion = ("                if os.environ.get('DT_ACCEPTED_STORAGE_PROFILE') == " + repr(MODE) + ':\n'
                 '                    from accepted_dt_storage_profile import configure_existing_producer\n'
                 '                    configure_existing_producer(self)\n')
    assert before.count(anchor) == 1
    after = before.replace(anchor, '                )\n' + insertion +
                           '            return self._deltatrace_producer.attribute_prepared_batch(data)')
    compile(after, str(worker), 'exec')

    class RemoveAddedInitialization(ast.NodeTransformer):
        removed = 0

        def visit_If(self, node):
            if ast.unparse(node.test) == "os.environ.get('DT_ACCEPTED_STORAGE_PROFILE') == " + repr(MODE):
                self.removed += 1
                return None
            return self.generic_visit(node)

    remove = RemoveAddedInitialization()
    default_tree = remove.visit(ast.parse(after))
    assert remove.removed == 1
    assert ast.dump(default_tree, include_attributes=False) == ast.dump(ast.parse(before), include_attributes=False)
    temporary = worker.with_name(worker.name + '.resource-profile-tmp')
    temporary.write_text(after)
    os.replace(temporary, worker)  # Break the hardlink; preserve the failed execution source.
    assert sha(parent) == '198f509c9a0af8a03d54f22e66f5f0d5593ab4c9eb7b27777c201ef323d2fecb'
    expected = {
        CANDIDATE / 'entry/reward_readout.py': '812ef4662771055d1045d78ed76cc8a1d2d8bd3ae607d1b6e9ec809a49ee51cd',
        CANDIDATE / 'entry/dt_training_batch.py': '006419cd4c186a44f4bbbc895994eff032e01136f5e70d9036915750266dcfc4',
        target / 'verl/trainer/ppo/ray_trainer.py': '9d098e3874d2579b885935a66a60285e7d419250b34e4c79d6a3efd7c329c13e',
        CANDIDATE / 'entry/textcraft_owner_rollout.py': 'da35841f5319a427f079bb6f6a297dce741157e1d4fba3c20f0577de193fd7b9',
    }
    assert all(sha(path) == digest for path, digest in expected.items())
    bindings = {str(path): digest for path, digest in expected.items()}
    bindings.update({str(worker): sha(worker), str(bridge): sha(bridge)})
    record = dict(status='prepared_only_native_producer_resource_initialization',
                  prepared_unix=time.time(), candidate=str(CANDIDATE),
                  source_bindings=bindings, storage_mode=MODE,
                  parent_worker_SHA256=sha(parent),
                  default_worker_AST_unchanged_when_added_branch_removed=True,
                  original_v10_credit_generator_and_trainer_bytes_unchanged=True,
                  model_calls=0, DT_calls=0, optimizer_calls=0, training_launches=0,
                  native_checkpoint_restore_executed=False,
                  next='Bind explicit same-run checkpoint5 resume and original resume-chain reader; native CLI verification before any launch.')
    (CANDIDATE / 'resource-source-prepared.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()

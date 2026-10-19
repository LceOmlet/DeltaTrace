"""Compare the native per-handler prompt API with HF's existing batch API.

Only saved official TextCraft conversations and the provisioned tokenizer are
read. No environment, model, DT, optimizer or CUDA call is made.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import time
from types import SimpleNamespace


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from verl.utils import hf_tokenizer

    root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    audit = root / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
    owner_path = root / 'runs/official-trajectory-20260930-v4/textcraft-owner/AgentGym-RL/verl/workers/rollout/schemas.py'
    owner_sha = hashlib.sha256(owner_path.read_bytes()).hexdigest()
    assert owner_sha == '6b479fbc9ffa3689a7a89b51ac8fb555b9877c2f81a558286da5bff5dc383e46'
    spec = importlib.util.spec_from_file_location('native_prompt_schema_CPU_check', owner_path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    trajectory_path = root / 'saves/test/executer_logs/step40/0.json'
    if not trajectory_path.exists():
        # The owner writes relative to the original TaskRunner working directory.
        import psutil
        trajectory_path = Path(psutil.Process(2507756).cwd()) / 'saves/test/executer_logs/step40/0.json'
    trajectories = json.loads(trajectory_path.read_bytes())
    model_path = '/mnt/si0021787ci2/default/models/Qwen3.5-9B'
    tokenizer = hf_tokenizer(model_path, trust_remote_code=True, local_files_only=True)
    examples = {}
    for turn in (0, 7, 15):
        conversations = []
        for row in trajectories:
            messages = row['conversations']
            positions = [i for i, message in enumerate(messages) if message['role'] == 'assistant']
            if len(positions) > turn:
                conversations.append(messages[:positions[turn]])
        if conversations:
            examples['actual_turn_' + str(turn + 1)] = conversations
    first = examples['actual_turn_1']
    for size in (0, 1, 2, 8, 32):
        examples['actual_prefix_rows_' + str(size)] = first[:size]
    report = dict(status='running', started=time.time(), tokenizer_class=type(tokenizer).__name__,
                  tokenizer_model=model_path, native_schema=str(owner_path), native_schema_SHA256=owner_sha,
                  real_trajectory_source=str(trajectory_path),
                  real_trajectory_source_SHA256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
                  cases=[], additional_model_calls=0, additional_DT_calls=0,
                  additional_environment_calls=0, additional_optimizer_calls=0,
                  production_changes=0)
    for label, conversations in examples.items():
        handlers = [SimpleNamespace(messages=[owner.Message(**message) for message in conv])
                    for conv in conversations]
        stamp = time.perf_counter()
        original = [owner.RolloutHandler.get_generation_prompt(h, tokenizer) for h in handlers]
        original_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        batched = (tokenizer.apply_chat_template(conversations, add_generation_prompt=True,
                   tokenize=True, return_dict=False) if conversations else [])
        batch_seconds = time.perf_counter() - stamp
        assert original == batched, label
        stamp = time.perf_counter()
        warm_original = [owner.RolloutHandler.get_generation_prompt(h, tokenizer) for h in handlers]
        warm_seconds = time.perf_counter() - stamp
        assert original == warm_original, label
        report['cases'].append(dict(label=label, rows=len(conversations),
            tokens=sum(map(len, original)), exact_token_ID_match=True,
            original_seconds=original_seconds, batch_seconds=batch_seconds,
            warm_original_seconds=warm_seconds))
    report.update(status='completed_CPU_prompt_batch_interface_check', finished=time.time(),
                  CUDA_initialized=torch.cuda.is_initialized(),
                  production_deployed=False,
                  scope='Integer prompt-ID representation equivalence and CPU timing only. Not a floating-point model/FA/FLA/VERL tolerance, generation-stage speedup or GPU-Util claim.')
    target = audit / 'native-chat-template-batch-CPU-results.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

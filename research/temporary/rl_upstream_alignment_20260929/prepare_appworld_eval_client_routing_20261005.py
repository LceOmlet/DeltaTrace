"""Isolated IPC fix: retain LOOP's chosen eval client through native RPC.

This only patches the three frozen local integration sources. LOOP's sampler,
submission/round-robin method, request construction, rewards and train settings
are not rewritten. No remote action or model import is performed.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "loop_owner_rollout.py": "c4a46492a665dbb6b1ac9076a6309ed3e128cb534068fca4f65c9f9a5532fcce",
    "loop_owner_worker.py": "6ad5a3e383d032fdc7f0dd2dab1ee027d8f0ed182f9725d07a3faedb46e54ec9",
    "loop_async_transport.py": "3d6438314908dba0076d0d1874025f321e7627d5f63193801e5b70165ea7ecc1",
}


def _replace(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(f"Expected one source seam: {before!r}")
    return source.replace(before, after, 1)


def patched_sources(originals: dict[str, bytes]) -> dict[str, bytes]:
    if set(originals) != set(EXPECTED):
        raise ValueError("Exactly the three frozen integration files are required")
    for name, digest in EXPECTED.items():
        actual = hashlib.sha256(originals[name]).hexdigest()
        if actual != digest:
            raise ValueError(f"Frozen source mismatch {name}: {actual}")
    sources = {name: content.decode("utf-8") for name, content in originals.items()}
    name = "loop_owner_rollout.py"
    sources[name] = _replace(sources[name],
        "            self.processes = OwnerProcesses(OmegaConf.to_container(self.native, resolve=True),\n"
        "                self.config.actor_rollout_ref.model.path, self.reserve, self.evaluation, world)",
        "            inference_clients = 1\n"
        "            if self.evaluation and hasattr(collector, 'async_rollout_manager'):\n"
        "                inference_clients = len(collector.async_rollout_manager.async_llm_servers)\n"
        "            self.processes = OwnerProcesses(OmegaConf.to_container(self.native, resolve=True),\n"
        "                self.config.actor_rollout_ref.model.path, self.reserve, self.evaluation, world,\n"
        "                inference_clients)")
    name = "loop_owner_worker.py"
    sources[name] = _replace(sources[name], "import uuid\n", "import uuid\nfrom functools import partial\n")
    sources[name] = _replace(sources[name],
        "              evaluation, commands, replies, output, cancellation):",
        "              evaluation, commands, replies, output, cancellation, inference_clients=1):")
    sources[name] = _replace(sources[name], "        def completion(request):", "        def completion(request, *, server_index=None):")
    sources[name] = _replace(sources[name],
        "            output.put(('completion', rank, key, request))",
        "            event = ('completion', rank, key, request)\n"
        "            if server_index is not None:\n"
        "                event += (server_index,)\n"
        "            output.put(event)")
    sources[name] = _replace(sources[name],
        "        template._vllm._completion_transport = completion",
        "        template._vllm._completion_transport = completion\n"
        "        eval_clients = [template]\n"
        "        if evaluation:\n"
        "            if inference_clients < 1:\n"
        "                raise ValueError('Evaluation requires at least one native server')\n"
        "            for _ in range(1, inference_clients):\n"
        "                eval_clients.append(VLLMQwen3(host='127.0.0.1', port=0, base_model_path=Path(model_path),\n"
        "                    model_id=None, temperature=cfg.llm.temperature,\n"
        "                    max_model_len=cfg.llm.vllm_server.max_model_len-reserve, **client_config))\n"
        "            if inference_clients > 1:\n"
        "                for server_index, client in enumerate(eval_clients):\n"
        "                    client._vllm._completion_transport = partial(completion, server_index=server_index)")
    sources[name] = _replace(sources[name],
        "        def clients(event):\n"
        "            template._vllm._cancellation_event = event\n"
        "            return [template]",
        "        def clients(event):\n"
        "            if evaluation:\n"
        "                for client in eval_clients:\n"
        "                    client._vllm._cancellation_event = event\n"
        "                return eval_clients\n"
        "            template._vllm._cancellation_event = event\n"
        "            return [template]")
    sources[name] = _replace(sources[name],
        "    def __init__(self, config, model_path, reserve, evaluation, world_size):",
        "    def __init__(self, config, model_path, reserve, evaluation, world_size, inference_clients=1):")
    sources[name] = _replace(sources[name],
        "            self.output, self.cancellations[rank])) for rank in range(world_size)]",
        "            self.output, self.cancellations[rank], inference_clients)) for rank in range(world_size)]")
    name = "loop_async_transport.py"
    sources[name] = _replace(sources[name], "            _,rank,key,raw=event",
        "            _,rank,key,raw=event[:4]\n"
        "            server_index=event[4] if len(event)==5 else rank")
    sources[name] = _replace(sources[name],
        "            ref=servers[rank].generate_tokens.remote(ids,options,key)\n"
        "            inflight[key]=(rank,prompt,ref)",
        "            ref=servers[server_index].generate_tokens.remote(ids,options,key)\n"
        "            inflight[key]=(rank,prompt,ref,server_index)")
    sources[name] = _replace(sources[name],
        "            key=event[1];rank,prompt,ref=inflight.pop(key)",
        "            key=event[1];rank,prompt,ref,server_index=inflight.pop(key)")
    sources[name] = _replace(sources[name],
        "                  f'elapsed={time.monotonic()-started:.1f}s completed_ranks={len(results)}',flush=True)",
        "                  f'elapsed={time.monotonic()-started:.1f}s completed_ranks={len(results)} '\n"
        "                  f'server_index={server_index} reply_rank={rank}',flush=True)")
    sources[name] = _replace(sources[name],
        "        for key,(rank,_,_) in inflight.items():",
        "        for key,(rank,_,_,server_index) in inflight.items():")
    sources[name] = _replace(sources[name],
        "                servers[rank].abort_request.remote(key);aborted.add(key)",
        "                servers[server_index].abort_request.remote(key);aborted.add(key)")
    result = {name: source.encode("utf-8") for name, source in sources.items()}
    for name, content in result.items():
        ast.parse(content, filename=name)
    return result


def prepare(baseline: Path, candidate: Path) -> dict:
    originals = {name: (baseline / name).read_bytes() for name in EXPECTED}
    patched = patched_sources(originals)
    candidate.mkdir(parents=True, exist_ok=False)
    receipt = {"state": "prepared-only", "scope": "eval client identity IPC only",
        "training_default_unchanged": True, "official_selection_reimplemented": False,
        "numerical_acceptance": "not performed; CPU dispatch is not GPU tolerance evidence",
        "baseline": str(baseline.resolve()), "candidate": str(candidate.resolve()), "files": {}}
    try:
        for name, content in patched.items():
            (candidate / name).write_bytes(content)
            (candidate / (name + ".diff")).write_text("".join(difflib.unified_diff(
                originals[name].decode("utf-8").splitlines(keepends=True),
                content.decode("utf-8").splitlines(keepends=True),
                fromfile="baseline/" + name, tofile="candidate/" + name)), encoding="utf-8")
            receipt["files"][name] = {"original_sha256": EXPECTED[name],
                "candidate_sha256": hashlib.sha256(content).hexdigest()}
    except BaseException as exc:
        receipt["state"] = "preparation-failed"
        receipt["error"] = repr(exc)
        raise
    finally:
        (candidate / "prepared.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline, args.candidate), indent=2))


if __name__ == "__main__":
    main()

"""Release unused native pinned storage after a complete owner compute phase.

This is an opt-in resource patch for the observed MetaX host OOM.  It invokes
the installed PyTorch allocator, without implementing an allocator, changing
offload policy, or flushing between microbatches.  The owner algorithms and
their default path remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
import textwrap
from pathlib import Path


ENABLE_ENV = "VERL_RELEASE_UNUSED_HOST_CACHE"
METHODS = ("update_actor", "compute_dt_token_advantages")


def release_statement(phase: str) -> str:
    return textwrap.dedent(f'''\
        if os.environ.get("{ENABLE_ENV}") == "1":
            host_reserved_before = torch.cuda.memory.host_memory_stats()["reserved_bytes.current"]
            with Timer(name="release_unused_host_cache", logger=None) as host_cache_timer:
                torch._C._host_emptyCache()
            host_reserved_after = torch.cuda.memory.host_memory_stats()["reserved_bytes.current"]
            print(f"[native_host_cache] phase={phase} reserved_before={{host_reserved_before}} reserved_after={{host_reserved_after}} release_s={{host_cache_timer.last:.6f}}", flush=True)
    ''')


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    methods = {node.name: node for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name in METHODS}
    if set(methods) != set(METHODS):
        raise RuntimeError("The pinned owner compute boundaries are missing")
    edits = []
    lines = source.splitlines(keepends=True)
    for name, method in methods.items():
        existing = [node for node in ast.walk(method) if isinstance(node, ast.If)
                    and ENABLE_ENV in ast.unparse(node.test)]
        if existing:
            if len(existing) != 1 or ast.dump(existing[0], include_attributes=False) != ast.dump(
                    ast.parse(release_statement(name)).body[0], include_attributes=False):
                raise RuntimeError(f"Different host-cache handling already exists in {name}")
            continue
        if name == "update_actor":
            boundary = method.body[-1]
            if not isinstance(boundary, ast.Return) or ast.unparse(boundary.value) != "output":
                raise RuntimeError("The original update_actor return boundary has changed")
            line, indentation = boundary.lineno - 1, boundary.col_offset
        else:
            boundary = method.body[-1]
            if not isinstance(boundary, ast.Try) or not boundary.finalbody:
                raise RuntimeError("The original DT finally/offload boundary has changed")
            line, indentation = boundary.finalbody[-1].end_lineno, boundary.finalbody[-1].col_offset
        edits.append((line, textwrap.indent(release_statement(name), " " * indentation)))
    for line, addition in sorted(edits, reverse=True):
        lines.insert(line, addition)
    patched = "".join(lines)
    ast.parse(patched)
    return patched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner_file", type=Path)
    args = parser.parse_args()
    original = args.owner_file.read_text()
    patched = patch_source(original)
    if patched != original:
        args.owner_file.write_text(patched)

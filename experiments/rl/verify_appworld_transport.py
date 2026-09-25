"""Bounded reproduction against an existing, unused official AppWorld server.

Uses the original execute implementation and the patched original HTTP helper;
this is a transport regression check, not task-performance evaluation.
"""
import argparse
import ast
import inspect
import json
from pathlib import Path
import time
from unittest.mock import patch

import requests
import appworld.environment as owner
from patch_appworld_transport import NEW, OLD, patch_source


def helper(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "AppWorld")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_remote_environment_call_helper")
    node.decorator_list = []
    scope = dict(vars(owner))
    exec(compile(ast.Module(body=[node], type_ignores=[]), owner.__file__, "exec"), scope)
    return scope[node.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(owner.__file__).read_text().replace(NEW, OLD)
    candidate = patch_source(source)
    assert patch_source(candidate) == candidate
    result = {"owner_file": owner.__file__, "url": args.url, "cases": []}
    for name, text in (("original", source), ("candidate", candidate)):
        with patch.object(owner.AppWorld, "_remote_environment_call_helper", staticmethod(helper(text))):
            world = owner.AppWorld(task_id="b7a9ee9_1", experiment_name=f"dt_transport_check_{name}",
                                   remote_environment_url=args.url)
            # The same official initializer sets a short server execution limit.
            # Initialization itself retains the normal transport deadline.
            world._remote_environment_call("initialize", task_id=world.task_id,
                                          experiment_name=f"dt_transport_check_{name}", timeout_seconds=1)
            world.timeout_seconds = 1
            normal = world.execute("print(6 * 7)")
            before = world.num_interactions
            started = time.perf_counter()
            try:
                observation = world.execute("while True:\n    page_playlists = apis.spotify.show_playlist_library")
                error = None
            except requests.exceptions.ReadTimeout:
                observation, error = None, "ReadTimeout"
            case = dict(path=name, seconds=time.perf_counter()-started, normal=normal,
                        observation=observation, error=error,
                        client_interactions_added=world.num_interactions-before)
            # Server may finish writing its original timeout log after old-client failure.
            world.timeout_seconds = 100
            case["next_observation"] = world.execute("print(7 * 7)")
            case["task_completed"] = world.task_completed()
            case["official_success"] = world.evaluate().success
            world.close()
            result["cases"].append(case)
            print(json.dumps(case), flush=True)
    old, new = result["cases"]
    assert old["error"] == "ReadTimeout", old
    assert new["error"] is None and "timed out after 1 seconds" in new["observation"], new
    assert new["client_interactions_added"] == 1
    assert old["normal"] == new["normal"] == "42\n"
    assert old["next_observation"] == new["next_observation"] == "49\n"
    assert old["official_success"] == new["official_success"]
    result["status"] = "passed"
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()

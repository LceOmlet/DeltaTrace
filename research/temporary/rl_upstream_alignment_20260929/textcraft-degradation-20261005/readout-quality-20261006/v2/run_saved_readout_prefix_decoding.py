"""Reuse the frozen CPU environment wrapper solely for original-ID decoding."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

LOCAL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_b4_cpu_runner", LOCAL / "run_saved_native_b4_denominators.py")
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def main():
    script_name = "decode_saved_readout_prefixes.py"
    result_name = "native-prefix-decoding.json"
    receipt_name = "native-prefix-decoding-cpu-run.json"
    log_name = "native-prefix-decoding-cpu.stdout.txt"
    script_path = LOCAL / script_name
    ast.parse(script_path.read_text(encoding="utf-8"))
    expected = hashlib.sha256(script_path.read_bytes()).hexdigest()
    subprocess.run(owner.stage.SCP + [str(script_path), f"{owner.stage.SSH[-1]}:{owner.OUT}/{script_name}"], check=True)
    remote = owner.SCRIPT.replace("CPU saved original DataProto analysis only", "CPU original saved IDs tokenizer decoding only")
    for key, value in {"NATIVE": owner.NATIVE, "OUT": owner.OUT, "HASH": expected,
                       "ANALYZER": script_name, "RESULT": result_name,
                       "RECEIPT": receipt_name, "LOG": log_name}.items():
        remote = remote.replace("__" + key + "__", value)
    result = subprocess.run(owner.stage.SSH + ["bash", "-s"], input=remote.encode(), capture_output=True)
    (LOCAL / "native-prefix-decoding-ssh.stdout.txt").write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors="replace")); print(result.stderr.decode(errors="replace"))
    for name in (result_name, receipt_name, log_name):
        subprocess.run(owner.stage.SCP + [f"{owner.stage.SSH[-1]}:{owner.OUT}/{name}", str(LOCAL / name)], check=True)
    binding = {"scope": "Exact decode-only fetch; reused prior frozen CPU environment wrapper without modifying it",
        "launcher": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "CPU_environment_wrapper": {"path": str(LOCAL / "run_saved_native_b4_denominators.py"),
            "sha256": hashlib.sha256((LOCAL / "run_saved_native_b4_denominators.py").read_bytes()).hexdigest()},
        "files": [{"path": str(LOCAL / name), "sha256": hashlib.sha256((LOCAL / name).read_bytes()).hexdigest(),
                   "bytes": (LOCAL / name).stat().st_size} for name in (result_name, receipt_name, log_name)]}
    (LOCAL / "native-prefix-decoding-local-fetch.json").write_text(json.dumps(binding, indent=2) + "\n")
    result.check_returncode()


if __name__ == "__main__":
    main()

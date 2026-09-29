"""Run one local test group with a Windows memory limit and resource receipt.

This is a test launcher only. It never starts remote jobs or training services.
Use the existing CPU-capable Python environment; dependencies are not installed
by this script. The Windows job contains only the child launched here.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def memory_job(limit_bytes: int):
    if os.name != "nt":
        raise RuntimeError("This local resource launcher targets Windows")
    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]
    class Counters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Counters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    api.CreateJobObjectW.restype = wintypes.HANDLE
    api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = Extended()
    # JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    limits.BasicLimitInformation.LimitFlags = 0x200 | 0x2000
    limits.JobMemoryLimit = limit_bytes
    if not api.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        api.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    return api, handle


def gpu_memory():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=10)
        return {"exit_code": p.returncode, "physical_gpu_memory_mib": p.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--memory-gib", type=float, default=4)
    parser.add_argument("--min-free-gib", type=float, default=6)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a local command is required after --")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.receipt.with_suffix(".log")
    floor = int(args.min_free_gib * 2**30)
    if psutil.virtual_memory().available < floor:
        raise RuntimeError("Available host memory is below the test safety floor")
    # Windows/CUDA did not hide the device with an empty value on this host.
    # -1 is an explicit invalid device and was verified with torch.cuda.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
    record = {"command": command, "python": sys.executable,
              "job_commit_limit_bytes": int(args.memory_gib * 2**30),
              "minimum_available_bytes": floor, "cuda_visible_devices": "-1",
              "threads": 2, "gpu_before": gpu_memory(), "samples": []}
    api, job = memory_job(record["job_commit_limit_bytes"])
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if not api.AssignProcessToJobObject(job, wintypes.HANDLE(child._handle)):
                child.kill()
                child.wait()
                raise ctypes.WinError(ctypes.get_last_error())
            proc = psutil.Process(child.pid)
            while child.poll() is None:
                rss = 0
                try:
                    processes = [proc] + proc.children(recursive=True)
                except psutil.NoSuchProcess:
                    processes = []
                for item in processes:
                    try:
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                available = psutil.virtual_memory().available
                elapsed = time.monotonic() - started
                record["samples"].append({"seconds": elapsed, "tree_rss_bytes": rss,
                                          "host_available_bytes": available})
                if available < floor or elapsed > args.timeout:
                    record["stopped_reason"] = "host_memory_floor" if available < floor else "timeout"
                    api.TerminateJobObject(job, 124)
                    break
                time.sleep(.25)
            record["exit_code"] = child.wait()
    finally:
        api.CloseHandle(job)
        record["seconds"] = time.monotonic() - started
        record["peak_sampled_tree_rss_bytes"] = max((x["tree_rss_bytes"] for x in record["samples"]), default=0)
        record["gpu_after"] = gpu_memory()
        record["log"] = str(log_path.resolve())
        args.receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(log_path.read_text(encoding="utf-8"))
    print(json.dumps({k: v for k, v in record.items() if k != "samples"}, indent=2))
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

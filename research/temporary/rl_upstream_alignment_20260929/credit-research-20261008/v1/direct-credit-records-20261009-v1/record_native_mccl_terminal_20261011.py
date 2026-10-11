"""Bind completed incident replay, actual owner identities, and evidence limits."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path.cwd()
ARTIFACTS = ROOT / "native-MCCL-collective-log-terminal-artifacts-20261011-v1"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def identity(path):
    data = path.read_bytes()
    return dict(path=path.relative_to(WORKSPACE).as_posix(), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())

manifest = read(ARTIFACTS / "manifest.json")
launch = read(WORKSPACE / "experiments/rl/results_textcraft_incident37_MCCL_collective_log_launch_20261011.json")
recheck = read(ROOT / "native-MCCL-terminal-and-owner-recheck-20261011-v1.json")
comparison = read(ROOT / "native-FSDP-Python-owner-comparison-20261011-v1.json")
communication = read(ROOT / "native-MCCL-complete-record-analysis-20261011-v1.json")
imports = read(ROOT / "native-DT-storage-actual-imports-20261011-v1.json")
capture = read(ROOT / "native-DT-storage-capture-namespace-20261011-v1.json")
assert manifest["all_terminal"] and all(p.get("missing") for p in recheck["processes"])
assert recheck["source_sha256"] == launch["launch"]["formal_source_sha256"]
assert all(x["byte_equal"] for x in comparison["files"])
current_hashes = {Path(x["path"]).name: x["sha256"] for x in recheck["installed_files"]}
assert all(current_hashes[x["name"]] == x["official_sha256"] for x in comparison["files"])
configuration = identity(ARTIFACTS / "result/initialization-config.yaml")
assert configuration["sha256"] == "2b17044560b2b6de97cdba5fc6b3223195ba2cefe496e087e6072d769281e448"

ranks = []
for rank in manifest["ranks"]:
    phases = [json.loads(line) for line in (ARTIFACTS / f"result/rank{rank['rank']}-phases.jsonl").read_text(encoding="utf-8").splitlines()]
    times = {x["phase"]: x["unix"] for x in phases}
    assert len(rank["microbatches"]) == 8 and all(x["gradient_finite"] for x in rank["microbatches"])
    assert rank["actual_optimizer_steps"] == 1
    ranks.append(dict(rank=rank["rank"], pid=rank["pid"], birth=rank["birth"], terminal=rank["phase"],
                      finite_B4=8, original_diagnostic_optimizer_steps=1,
                      DT_seconds=times["native_saved_DT_batch_completed"]-times["native_saved_DT_batch_begin"],
                      actor_seconds=times["complete"]-times["native_update_policy_begin"]))

credit = next(x for x in recheck["DT_seams"] if x["path"].endswith("deltatrace_credit.py"))
tree = ast.parse(credit["text"])
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "trace_token_attribution")
finalizers = [ast.unparse(n) for n in ast.walk(function) if isinstance(n, ast.Try) and n.finalbody]
assert any("release_owner_params" in t for t in finalizers)
receipt = dict(
    observed_unix=time.time(),
    status="Native MCCL-log replay terminal finite; five FSDP Python owners byte-identical to PyTorch 2.8.0; faulty writer still unknown",
    recording_parent_commit=subprocess.check_output(["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"], text=True).strip(),
    previous_goal_turn_classification="progress",
    previous_goal_turn_basis="The preceding goal continuation collected the original native MCCL diagnostic's terminal artifacts and exposed the literal shared log filename, changing the address-analysis scope. The intervening direct-user answer only verified the existing incident and displayed its curves.",
    formal=dict(task="TextCraft", stopped=True, iterations=37, budget=330, restarted=False, restored_checkpoint=False),
    production_changes=0, method_changes=0, new_tolerance=False,
    numerical_source_commit="26bef6c8b2e49db118f46e3c05e86944dcf8e293",
    accepted_GDN_sha256="7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656",
    upstream_commit="20bd331bdbc9026a5668e11362178e10ab7400c8",
    source_sha256=recheck["source_sha256"], configuration=configuration,
    fixed_parameters=dict(lora_rank=8, lora_alpha=16, actor_microbatch_per_card=4, effective_two_card_microbatch=8),
    original_diagnostic_chain=launch["launch"]["original_binding"],
    driver=dict(pid=4081870, birth=1791704016.23, terminal=True),
    diagnostic_physical_GPUs=[4, 5], ranks=ranks,
    diagnostic_scope="Saved incident trainable-state replay only; one joint original optimizer update across two ranks, not a formal restart or two production updates.",
    captured_weight_rows=manifest["GEMM_rows"],
    native_MCCL=dict(environment=launch["launch"]["native_diagnostic_environment"],
                     raw_log=identity(ARTIFACTS / "mccl.%h.%p.log"), analysis=identity(ROOT / "native-MCCL-complete-record-analysis-20261011-v1.json"),
                     complete_messages=communication["complete_records"],
                     filename_expansion_supported=False,
                     limitation="Two processes wrote the literal filename and overwrote/interleaved records. No fault reproduced; negative address matches do not exclude overlap."),
    owner_comparison=dict(receipt=identity(ROOT / "native-FSDP-Python-owner-comparison-20261011-v1.json"),
                          verified_fresh_against_installed=True, files=comparison["files"], scope=comparison["scope"]),
    DT_lifecycle_review=dict(
        actual_seams=[{k:v for k,v in row.items() if k != "text"} for row in recheck["DT_seams"]],
        actual_imports=identity(ROOT / "native-DT-storage-actual-imports-20261011-v1.json"),
        namespace=identity(ROOT / "native-DT-storage-capture-namespace-20261011-v1.json"),
        import_inspection_error="accelerated.qwen35 is a namespace with __file__=None; the first inspection incorrectly treated it as a file. Corrected through its actual capture binding; no runtime import defect.",
        release_finalizer=finalizers,
        precision_scope="native_fla_fp16 casts FLA operands and restores the operator binding; no model weight assignment in that module.",
        conclusion="No direct parameter write identified in these reviewed seams. Public unshard/reshard calls and Python owner identity do not prove the complete storage lifecycle safe.",
        CUDA_initialized_during_CPU_imports=False,
        CPU_import_peak_RSS_bytes=[imports["peak_RSS_bytes"], capture["peak_RSS_bytes"]],
    ),
    fresh_recheck=identity(ROOT / "native-MCCL-terminal-and-owner-recheck-20261011-v1.json"),
    process_recheck=recheck["processes"], host_available_bytes=recheck["host_available"],
    physical_gpu_after_terminal=recheck["physical"],
    no_new_GPU_replay_in_this_continuation=True,
    limitation="Frozen head row corruption and subsequent KL backward NaN remain unresolved. Normal diagnostic outputs are not a production repair or numerical acceptance.",
    next_required_evidence="Identify the actual faulty storage write/free/reuse in a failing execution; do not infer it from incomplete normal logs or change PPO/credit to hide the NaN.",
)
p = WORKSPACE / "experiments/rl/results_textcraft_incident37_MCCL_terminal_owner_20261011.json"
p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

snapshot = WORKSPACE / "experiments/rl/current_runtime.json"
before = snapshot.read_bytes()
previous = json.loads(before)
key = "latest_textcraft_incident37_MCCL_terminal_owner_20261011"
assert key not in previous
entry = dict(receipt=identity(p), status=receipt["status"], observed_unix=receipt["observed_unix"],
             formal_iterations=37, formal_budget=330, formal_stopped=True,
             production_changes=0, formal_restarted=False, diagnostic_all_terminal=True,
             installed_FSDP_Python_files_equal_official=5, faulty_writer_identified=False)
nl = b"\r\n" if before.startswith(b"{\r\n") else b"\n"
tail = before[1+len(nl):]
prefix = json.dumps({key: entry}, ensure_ascii=False, indent=2)[:-2].encode().replace(b"\n", nl)
after = prefix + b"," + nl + tail
new = json.loads(after)
assert all(new[k] == v for k, v in previous.items())
snapshot.write_bytes(after)

runtime = WORKSPACE / "experiments/rl/RUNTIME_RECORD.md"
before = runtime.read_bytes()
first_end = before.index(b"\n") + 1
section = """
## 2026-10-11 MCCL诊断已结束；原FSDP Python与官方逐字节一致

原driver4081870及actor4086511/4088176均按出生时间复核已退出，两rank各8个B4
有限，共同一次原诊断optimizer；247处head行均为正常SHA4defedf。正式仍37/330
停止，无生产补丁、恢复检查点或重启，不把本次正常回放称为修复。

更正下方launch记录的“每PID日志”预期：本机安装的MetaX MCCL未展开%h/%p，
两进程写同一字面文件并有覆盖/交错。仅2818条字段完整记录可解析；不能据缺失
记录或本次未复现推断故障没有地址重叠。原日志及分析完整保留，未再启动GPU回放。

现场安装的FSDP collectives/param/param_group/state/fully_shard五文件，与PyTorch
v2.8.0原文件SHA逐一相同，远端再次读文件验证。该结论限于Python源码，不扩展到
MetaX C++、MCCL、allocator或实际GPU顺序。DT原接口通过public unshard/reshard、
trace finally释放；FLA精度适配只cast输入并恢复绑定，当前未找到直接权重写入。
accelerated.qwen35为namespace，__file__=None是合法行为；首个检查脚本误当文件，
已按实际capture绑定核对，不是训练导入错误。CPU导入峰RSS689MB以内，无CUDA。

最新物理八卡各858MiB、无进程，host可用1023459356672字节；这是终态资源，非峰值。
具体错误写入/释放/复用仍未定位；KL clamp不足以修复异常权重与错误log-prob。
见[终态、官方源码对照、实际接口与证据界限](results_textcraft_incident37_MCCL_terminal_owner_20261011.json)。

""".encode("utf-8")
runtime.write_bytes(before[:first_end] + section + before[first_end:])
print(json.dumps(dict(receipt=identity(p), runtime_key=key, formal_stopped=True,
                      production_changes=0, all_diagnostics_terminal=True), ensure_ascii=False))

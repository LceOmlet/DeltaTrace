"""Record completed v3 receipts locally; no remote or training operations."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parents[2] / "experiments/rl"
assert not (RL / "results_textcraft_DTPO_variance_v3_20261015.json").exists(), "Preserve the recorded initial state and later progress."


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


prep = load("prepare_DTPO_variance_v3-dt.json")
regression = load("verify_DTPO_variance_v3-dt.json")
state = load("verify_DTPO_variance_v3_workers-dt.json")
registration = load("register_DTPO_variance_v3_workers-dt.json")
status = load("read_DTPO_variance_v3_status-dt.json")
stop = load("DTPO-variance-v3-previous-stop.json")
job = registration["report"]["job"]
root = job["run"].split("/runs/")[0]
candidate = root + "/candidates/" + job["version"]
owner = candidate + "/verl-owner"
assert state["report"]["complete"]
assert state["report"]["initial_optimizer0_scheduler0_verified"]
assert all(regression["report"]["checks"].values())
assert stop["status"] == "stopped_for_user_requested_fresh_v3"
assert not stop["survivors"]
assert status["report"]["alive"]
workers = state["report"]["workers"]
assert all((w["optimizer_states"], w["scheduler_last_epoch"]) == (0, 0) for w in workers)
files = [owner + "/" + s for s in (
    "verl/workers/actor/dp_actor.py", "verl/trainer/ppo/core_algos.py",
    "verl/workers/fsdp_workers.py", "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    "verl/trainer/ppo/ray_trainer.py", "verl/utils/torch_functional.py",
)] + [candidate + "/entry/dt_training_batch.py"]
bindings = {p: job["source_bindings"][p] for p in files}
assert bindings[owner + "/verl/trainer/ppo/ray_trainer.py"] == (
    "b4b0187ef747d1ffb91dbfb8d1b8f914f93d01d90ff58d24e386f2cb1b7eb943"
)
now = datetime.now(timezone(timedelta(hours=8))).isoformat()
record = {
    "recorded_at": now, "version": job["version"],
    "status": "deployed_actual_initial_workers_verified_first_iteration_pending",
    "job": {k: job[k] for k in (
        "pid", "birth", "started_unix", "run", "physical_gpus",
        "upstream_commit", "local_patch_commit",
    )},
    "canonical_method_patch_commit": "f726ee72", "source_bindings": bindings,
    "method": {
        "group_multiplier": "Pinned official VERL GRPO, original uid/traj_uid/reward/mask/std; no shadow implementation.",
        "DT_source": "g_Y * (-expm1(-d_i(Y)))", "DT_self_target": "g_Y",
        "observations_padding": "zero",
        "variance_only": prep["report"]["variance_standardization"],
        "PPO_reduction": "Original official token-mean; L=(1/16)*sum_j(S_j/N_j). Original actor/core unchanged.",
        "reduction_derivation": "Per-rank native optimizer minibatch32 / microbatch4 = 8 means; divide by8 then original two-rank data-parallel mean.",
        "no_global_token_denominator_candidate": True, "raw_environment_rewards_unchanged": True,
    },
    "fixed_configuration": prep["report"]["fixed_configuration"],
    "budget": prep["report"]["budget"],
    "initial_workers": [{k: w[k] for k in (
        "pid", "birth", "rank", "optimizer_states", "scheduler_last_epoch", "configuration",
    )} for w in workers],
    "actual_import_scope": "Original actor/core/FSDP/checkpoint files read through original worker RPC; trainer source frozen and bound by launch/source manifest. First actual v3 PPO input not yet observed.",
    "passive_records": state["report"]["passive_records"],
    "passive_recording_output_status": "mounted_on_both_actual_workers_first_DT_output_pending",
    "verification": {
        "real_saved_batch_CPU": {k: v for k, v in regression["report"].items() if k != "source_bindings"},
        "initial_state_receipt": state["receipt"],
        "manifest_registration_receipt": registration["receipt"],
        "scope": "Saved real-batch advantage boundary and actual initial native workers; no additional model/backward/optimizer calls. Not root-cause repair acceptance.",
    },
    "observation": {
        "unix": status["report"]["unix"], "completed_iterations": len(status["report"]["native_metrics"]),
        "total_iterations": 330,
        "phase": "first native generation / environment interaction; original log Rounds15/30 in progress",
        "driver_log": status["report"]["logs"]["driver.stderr.log"]["path"],
        "driver_lines": status["report"]["logs"]["driver.stderr.log"]["lines"],
        "metrics": status["report"]["native_metrics"], "receipt": status["receipt"],
        "physical_mx_smi": status["report"]["physical_mx_smi"],
        "native_cache_warning": {
            "path": status["report"]["read_errors"][0]["path"], "lines": [49, 54],
            "detail": "Inductor fx graph cache could not read another rank temporary .tmp file; warning logged, original generation continued. No cache clearing/reinstall performed.",
        },
    },
    "previous_DTPO": {
        "pid": stop["job"]["pid"], "birth": stop["job"]["birth"], "run": stop["job"]["run"],
        "status": stop["status"], "checkpoint_step_preserved": stop["checkpoint_step_preserved"],
        "checkpoint_directory": stop["checkpoint_directory"],
        "new_checkpoint_created": stop["new_checkpoint_created"], "survivors": stop["survivors"],
        "receipt": root + "/receipts/textcraft-gradient-rootcause-20261013-v1/DTPO-variance-v3-prepared/previous-DTPO-stop.json",
    },
    "protected": {
        "GRPO": {"pid": 3174448, "birth": 1791793629.73, "physical_gpus": [4, 5], "rerun": False},
        "PPO": {"pid": 2308539, "birth": 1791985613.77, "physical_gpus": [2, 3], "resumed_from_native_full_checkpoint": 15, "rerun": False},
        "other_project_physical_gpus": [6, 7],
    },
    "plot_policy": "Three-algorithm comparison using current DTPO v3 / original GRPO / original PPO; no splicing previous DTPO runs or missing-as-zero; auto y limits. No v3 completed iteration curve yet.",
    "rootcause_repair_verified": False, "goal_status": "active", "hourly_automation": "PAUSED",
    "full_remote_sources": {k: job["run"] + "/" + k + ".json" for k in ("launch", "source", "job")},
    "local_receipt_files": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in (
        "prepare_DTPO_variance_v3-dt.json", "verify_DTPO_variance_v3-dt.json",
        "verify_DTPO_variance_v3_workers-dt.json", "register_DTPO_variance_v3_workers-dt.json",
        "read_DTPO_variance_v3_status-dt.json", "DTPO-variance-v3-previous-stop.json",
        "DTPO-variance-v3-ray_trainer.py.diff",
    )},
}
output = RL / "results_textcraft_DTPO_variance_v3_20261015.json"
output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
text = f"""## {now} DTPO v3 已部署；原官方1/16归约保持，仅方差缩放不减均值

用户批准版本 `{job['version']}`，2026-10-14 23:59:31从基础权重重新启动，PID3140624/出生1791993570.92，physical0/1。上游20bd331b，方法/规范patch f726ee72，提交部署e05e23cb；新trainer b4b0187e、原DT入口a4e1f70b，实际原actor3a65e173/corefc2f992b/worker9dfd6ce7/checkpointf65b6538保持。原token-mean每卡32/4共8份均值、原双卡DP平均，即L=(1/16)*sum_j(S_j/N_j)，没有部署自编全局token分母候选。

保留固定官方GRPO组乘子与原DT组合，只在DT优势出口调用官方masked_var及原生torch.rsqrt(variance+1e-8)，不减均值。原masked_whiten(shift_mean=False)先减均值再加回原均值，并非本次A/std语义，不能直接替代。真实已保存256行CPU回归全部通过：零/padding不变、非零符号不变、Q/V/奖励/乘子不变；原masked std0.20124382、缩放4.969096、输出std约1且均值0.03544149保留。本项只验证边界接法，不把系数缩放当参数梯度倍率或根因已修复。

实际两个新worker3151306/1791993626.33、3153050/1791993643.98回读optimizer_states0/scheduler0，32k/LoRA8/16/B4/每rank32/global64/4更新/330/save5/keep2保持；原被动逐token记录已在两worker挂载，active2609d93f/disk814cfe92/GDN3f51f5f7实际核对，首DT输出待落盘。00:12:48原日志首轮生成/环境交互Rounds15/30中，0/330，尚无完整PPO更新。Inductor读另一rank临时缓存文件的警告原样保留，生成继续；没有清缓存或重装。完整实际路径、SHA、配置和回执见experiments/rl/results_textcraft_DTPO_variance_v3_20261015.json。

上一版no-whiten DTPO1419849已按本次重启要求停止，第5完整检查点保留，没有额外制造或删除检查点。原GRPO3174448/1791793629.73 physical4/5和已完整恢复的PPO2308539/1791985613.77 physical2/3继续，6/7其他项目保持。三算法同图入口已按ROOT当前manifest选择v3，不拼接旧DTPO；尚无v3完成点，不虚构曲线。小时检查仍PAUSED；grad norm加速增长根因goal保持active，首轮输入/更新和长期修复均未验收。

"""
for name in ("RUNTIME_RECORD.md", "REMOTE_ENVIRONMENT.md"):
    path = RL / name
    old = path.read_text(encoding="utf-8")
    assert job["version"] not in old[:5000], "Already recorded: inspect existing entry."
    path.write_text(text + old, encoding="utf-8")
path = RL / "current_runtime.json"
old = json.loads(path.read_text(encoding="utf-8"))
key = "20261015_DTPO_variance_v3_deployed_actual_initial"
assert key not in old, "Already recorded: inspect existing entry."
path.write_text(json.dumps({key: record, **old}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("saved", output, "recorded_at", now, "source_bindings", len(bindings))

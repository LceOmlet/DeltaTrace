"""Regress the observed cross-task session_latest plot contamination."""
from experiments.rl.plot_training_progress import parse_logs, read_job_logs


def test_missing_session_uses_each_tasks_own_driver_log(tmp_path):
    shared = tmp_path / "ray" / "session_latest" / "logs"
    shared.mkdir(parents=True)
    (shared / "worker-webshop.out").write_text(
        "[DT rollout] phase=dt_rpc_end seconds=121.493\n"
        "step:9 - training/global_step:9 - episode/success_rate:0.9\n")
    for task, step in (("Sokoban", "15/15"), ("AppWorld", "10/40")):
        log = tmp_path / f"{task}.log"
        log.write_text(f"[DT rollout] phase=generation_start step={step} active=4/4\n")
        records, errors = read_job_logs({"ray_tmpdir": str(tmp_path), "log": str(log)})
        metrics, phases = parse_logs(records)
        assert not metrics and not errors
        assert len(phases) == 1 and phases[0]["step"] == step
        assert phases[0]["source"]["path"] == str(log)


def test_recorded_session_preserves_direct_owner_output_without_double_count(tmp_path):
    session = tmp_path / "session-own"
    logs = session / "logs"
    logs.mkdir(parents=True)
    output = logs / "worker-task.out"
    text = "[DT rollout] phase=dt_rpc_end seconds=2.5\nstep:2 - training/global_step:2 - episode/success_rate:0.25\n"
    output.write_text(text)
    (logs / "worker-task.err").write_text("CUDA out of memory\n")
    driver = tmp_path / "train.log"
    driver.write_text(text)
    records, errors = read_job_logs({"ray_session": str(session), "log": str(driver)})
    metrics, phases = parse_logs(records)
    assert len(phases) == 1 and len(metrics) == 1
    assert len(metrics[0]["sources"]) == 1
    assert metrics[0]["values"]["episode/success_rate"] == 0.25
    assert phases[0]["source"]["path"] == str(output)
    assert len(errors) == 1 and errors[0]["path"].endswith("worker-task.err")

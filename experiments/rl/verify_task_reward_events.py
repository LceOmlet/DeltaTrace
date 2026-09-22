"""Check real task-worker rewards through the Q/V transport interface.

Uses scripted actions and explicit ratio fixtures. This is NOT a DT estimator,
model rollout, training run, or task-performance evaluation. Requires existing
task assets and tokenizer; never installs/downloads resources or starts servers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from counterfactual import reward_event_credit_for_episode


def task_rows(name, transitions, tokenizer):
    rows = []
    lengths = []
    history = []
    for step, (observation, action, next_obs, reward, done, info) in enumerate(transitions):
        history.append({"role": "user", "content": str(observation)})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True, return_dict=True,
        )["input_ids"]
        response = tokenizer.encode(action, add_special_tokens=False) + [tokenizer.eos_token_id]
        assert len(prompt) + len(response) <= 32768
        assert len(response) <= 512
        valid = len(response)
        padded = response + [tokenizer.pad_token_id] * (512 - valid)
        row = {
            "traj_uid": name, "uid": name, "env_step": step,
            "responses": torch.tensor(padded),
            "input_ids": torch.tensor(prompt + padded),
            "attention_mask": torch.tensor([1] * (len(prompt) + valid) + [0] * (512 - valid)),
            "active_masks": True, "rewards": float(reward),
            "dt_env_outcome": {"observation": next_obs, "info": info, "done": bool(done)},
        }
        rows.append(row)
        lengths.append({"prompt": len(prompt), "response": valid, "total": len(prompt) + valid})
        history.append({"role": "assistant", "content": action})
        if done:
            break
    return rows, lengths


def check_events(name, transitions, tokenizer):
    rows, lengths = task_rows(name, transitions, tokenizer)
    # Nonuniform values exercise transport without claiming a model estimate.
    ratios = [torch.linspace(-0.2, 0.3, 512).expand(len(rows), -1).clone() for _ in rows]
    credit = reward_event_credit_for_episode(rows, ratios)
    for index, result in enumerate(credit):
        future_return = sum(float(row["rewards"]) for row in rows[index:])
        assert abs(result["dt_q_estimates"][0].item() - future_return) < 1e-5
        mask = rows[index]["attention_mask"][-512:].bool()
        for value in result.values():
            assert not value[~mask].any()
        torch.testing.assert_close(
            result["dt_token_advantages"], result["dt_q_estimates"] - result["dt_v_estimates"],
            atol=2e-6, rtol=1e-5,
        )
    return {
        "events": len(rows), "rewards": [row["rewards"] for row in rows],
        "done": [row["dt_env_outcome"]["done"] for row in rows],
        "token_lengths_for_scripted_actions": lengths,
        "first_q_sample": float(credit[0]["dt_q_estimates"][0]),
        "transport_passed": True,
    }


def sokoban_events():
    from agent_system.environments.env_package.sokoban.envs import SokobanWorker
    from agent_system.environments.env_package.sokoban import sokoban_projection
    worker = SokobanWorker("tiny_rgb_array", dict(dim_room=(6, 6), num_boxes=1, max_steps=15, search_depth=30))
    obs, _ = worker.reset(7)
    transitions = []
    try:
        # Seed-7 official board, inspected once: includes an invalid-action
        # penalty, ordinary moves, and pushing its single box onto the goal.
        for move in ["still", "right", "up", "right", "down"]:
            text = f"<think>Reward interface check.</think><action>{move}</action>"
            actions, _ = sokoban_projection([text])
            nxt, reward, done, info = worker.step(actions[0])
            transitions.append((obs, text, nxt, reward, done, info))
            obs = nxt
            if done:
                break
        assert transitions[-1][4]
        assert transitions[-1][3] > 10
        assert all(abs(item[3] + 0.1) < 1e-6 for item in transitions[:-1])
        return transitions
    finally:
        worker.env.close()


def webshop_events():
    from agent_system.environments.env_package.webshop.envs import WebshopWorker
    from agent_system.environments.env_package.webshop import webshop_projection
    worker = WebshopWorker(7, dict(observation_mode="text", num_products=1000))
    obs, _ = worker.reset(500)
    transitions = []
    try:
        action = "search[shoes]"
        for step in range(3):
            text = f"<think>Reward interface check.</think><action>{action}</action>"
            actions, _ = webshop_projection([text])
            nxt, reward, done, info = worker.step(actions[0])
            transitions.append((obs, text, nxt, reward, done, info))
            obs = nxt
            if done:
                break
            available = worker.get_available_actions()["clickables"]
            if step == 0:
                products = [item for item in available if len(item) == 10 and item.lower().startswith("b")]
                assert products, available
                action = f"click[{products[0]}]"
            else:
                assert any(item.lower() == "buy now" for item in available), available
                action = "click[buy now]"
        assert transitions[-1][4]
        return transitions
    finally:
        worker.close()


def appworld_events():
    from appworld import AppWorld, load_task_ids
    from agent_system.environments.env_package.appworld.envs import AppWorldWorker
    from agent_system.environments.env_package.appworld import appworld_projection
    # Inject a real isolated local AppWorld into the real worker step method.
    # Existing HTTP servers and their active sessions remain untouched.
    worker = AppWorldWorker(worker_id=0, max_interactions=2, port=0)
    worker.env = AppWorld(
        task_id=load_task_ids("train")[0],
        experiment_name="dt_reward_event_interface_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"),
    )
    obs = worker.env.task.instruction
    transitions = []
    try:
        for code in ["help()", "print(1 + 1)"]:
            text = f"<think>Reward interface check.</think><code>{code}</code>"
            actions, _ = appworld_projection([text])
            nxt, reward, done, info = worker.step(actions[0])
            transitions.append((obs, text, nxt, reward, done, info))
            obs = nxt
            if done:
                break
        assert transitions[-1][4]
        return transitions
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"], local_files_only=True)
    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "context_cap": 32768, "model_loaded": False,
        "action_source": "scripted interface fixtures; not model-generated",
        "ratio_source": "synthetic interface fixtures; not DeltaTrace output",
        "training_verified": False, "dt_estimator_verified": False,
        "tasks": {},
    }
    for name, collect in [("sokoban", sokoban_events), ("webshop", webshop_events), ("appworld", appworld_events)]:
        try:
            results["tasks"][name] = check_events(name, collect(), tokenizer)
        except Exception as error:
            results["tasks"][name] = {"transport_passed": False, "error": repr(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(name, results["tasks"][name], flush=True)
    if not all(result["transport_passed"] for result in results["tasks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

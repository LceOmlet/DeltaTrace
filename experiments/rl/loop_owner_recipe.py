"""Read the selected author's command and compose it with the author's Hydra API.

This module does not run agents, score tasks, or implement a trainer. The pinned
repository has a README training command, rather than an AppWorld shell recipe.
Reading that command avoids a second hand-maintained list of experiment values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys


OWNER_COMMIT = "f14107a976e5793990329d3193df4742076c5a1d"


def training_command(owner_root: Path) -> list[str]:
    readme = (owner_root / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Model Training\n", 1)[1].split("\n## ", 1)[0]
    command = section.split("```bash\n", 1)[1].split("```", 1)[0]
    return shlex.split(command.replace("\\\n", " "))


def training_overrides(owner_root: Path) -> list[str]:
    command = training_command(owner_root)
    return command[command.index("./phi_agents/rl/train.py") + 1:]


def compose(owner_root: Path, overrides: list[str]):
    sys.path.insert(0, str(owner_root.resolve()))
    from phi_agents.rl.config import get_config

    return get_config("train", training_overrides(owner_root) + overrides)


def environment_configuration(owner_root: Path) -> dict:
    """Select native environment fields without reconstructing their defaults."""
    from omegaconf import OmegaConf

    cfg = compose(owner_root, [])
    evaluation = OmegaConf.merge(cfg, cfg.rl.eval.overrides)
    fields = {
        "training_environment": cfg.rl.scenario_runner,
        "training_task_sampler": cfg.rl.scenario_sampler,
        "evaluation_environment": evaluation.rl.scenario_runner,
        "evaluation_task_sampler": cfg.rl.eval.scenario_sampler,
    }
    result = {key: OmegaConf.to_container(value, resolve=True) for key, value in fields.items()}
    result["generation_boundary_reference"] = {
        "max_model_len": cfg.llm.vllm_server.max_model_len,
        "client": OmegaConf.to_container(cfg.llm.vllm_class, resolve=True),
        "training_temperature": cfg.llm.temperature,
        "evaluation_temperature": evaluation.llm.temperature,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "owner": "apple-aiml-research/ml-loop",
        "owner_commit": OWNER_COMMIT,
        "source": "README.md: Model Training",
        "readme_sha256": hashlib.sha256((args.owner_root / "README.md").read_bytes()).hexdigest(),
        "owner_overrides": training_overrides(args.owner_root),
        "environment": environment_configuration(args.owner_root),
        "scope": "Environment fields only, resolved by native Hydra. No trainer, optimizer, training budget, model or inference server is launched.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scope": result["scope"]}))


if __name__ == "__main__":
    main()

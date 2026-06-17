#!/usr/bin/env python3
"""Run evaluation on tool benchmark JSONL with a chosen agent backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.agents.tool_hf import ToolLocalHFAgent
from arb.agents.scripted import ToolExploitBaselineAgent, ToolHonestBaselineAgent
from arb.tool.factory import make_tool_environment
from arb.tool.metrics import aggregate_results
from arb.utils.io import read_jsonl


def run_episode(env, agent, max_turns: int = 20) -> dict:
    if hasattr(agent, "bind_task"):
        task_d = env.task.to_dict() if hasattr(env.task, "to_dict") else dict(env.task)
        agent.bind_task(task_d)

    history: list[dict[str, str]] = []
    prompt = env.get_prompt()
    obs: dict = {"done": False}

    for _ in range(max_turns):
        if obs.get("done"):
            break
        reply = agent.complete(prompt if not history else obs.get("observation", prompt), history)
        history.append({"role": "assistant", "content": reply})
        obs = env.step(reply)
        if not obs.get("done") and obs.get("observation"):
            history.append({"role": "system", "content": obs["observation"]})
            prompt = obs["observation"]

    if not obs.get("done"):
        obs = env.finalize_timeout()

    return {
        "task_id": env.task.id,
        "state": env.task.state,
        "success": obs.get("success", False),
        "metrics": obs.get("metrics", {}),
        "failure_reason": obs.get("failure_reason"),
        "transcript": env.transcript,
        "gold_aligned": obs.get("gold_aligned"),
        "tau_reward": obs.get("tau_reward"),
        "backend": obs.get("backend"),
        "exploit_detect_reason": obs.get("exploit_detect_reason"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on tool benchmark")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", default="results_tool.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--agent",
        default="honest_baseline",
        choices=["honest_baseline", "exploit_baseline", "local_hf"],
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()

    results = []
    tasks = list(read_jsonl(args.benchmark))
    if args.limit:
        tasks = tasks[: args.limit]

    agent = None
    if args.agent == "honest_baseline":
        pass
    elif args.agent == "exploit_baseline":
        pass
    else:
        if not args.model_path:
            raise ValueError("--model-path required for local_hf agent")
        agent = ToolLocalHFAgent(args.model_path)

    for task in tqdm(tasks, desc="tool_eval"):
        if args.agent == "honest_baseline":
            agent = ToolHonestBaselineAgent(task)
        elif args.agent == "exploit_baseline":
            agent = ToolExploitBaselineAgent(task)
        env = make_tool_environment(task)
        results.append(run_episode(env, agent, max_turns=args.max_turns))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate_results(results)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path} and {summary_path}")


if __name__ == "__main__":
    main()

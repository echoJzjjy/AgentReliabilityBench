#!/usr/bin/env python3
"""Run evaluation on text benchmark JSONL with a chosen agent backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.agents.local_hf import LocalHFAgent
from arb.agents.scripted import ExploitBaselineAgent, HonestBaselineAgent
from arb.text.environment import TextTaskEnvironment
from arb.text.metrics import aggregate_results
from arb.utils.io import read_jsonl


def run_episode(env: TextTaskEnvironment, agent, max_turns: int = 5) -> dict:
    history: list[dict[str, str]] = []
    obs: dict = {"done": False}

    for _ in range(max_turns):
        if obs.get("done"):
            break
        if not history:
            user_prompt = env.get_prompt()
        else:
            user_prompt = env.build_turn_prompt(obs.get("observation"))
        reply = agent.complete(user_prompt, history)
        history.append({"role": "assistant", "content": reply})
        obs = env.step(reply)

    out = {
        "task_id": env.task.id,
        "state": env.task.state,
        "success": obs.get("success", False),
        "metrics": obs.get("metrics", {}),
        "failure_reason": obs.get("failure_reason"),
        "transcript": env.transcript,
    }
    if obs.get("exploit_detect_reason"):
        out["exploit_detect_reason"] = obs["exploit_detect_reason"]
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on text benchmark")
    parser.add_argument("--benchmark", required=True, help="Path to *_solvable.jsonl etc.")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--agent",
        default="honest_baseline",
        choices=["honest_baseline", "exploit_baseline", "local_hf"],
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-turns", type=int, default=5)
    args = parser.parse_args()

    if args.agent == "honest_baseline":
        agent = HonestBaselineAgent()
    elif args.agent == "exploit_baseline":
        agent = ExploitBaselineAgent()
    else:
        if not args.model_path:
            raise ValueError("--model-path required for local_hf agent")
        agent = LocalHFAgent(args.model_path)

    results = []
    tasks = list(read_jsonl(args.benchmark))
    if args.limit:
        tasks = tasks[: args.limit]

    for task in tqdm(tasks, desc="eval"):
        env = TextTaskEnvironment(task)
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

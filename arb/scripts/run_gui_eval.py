#!/usr/bin/env python3
"""Run evaluation on GUI benchmark JSONL with a chosen agent backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arb.agents.local_hf import LocalHFAgent
from arb.agents.scripted import GuiExploitBaselineAgent, GuiHonestBaselineAgent
from arb.gui.environment import GuiTaskEnvironment
from arb.gui.metrics import aggregate_results
from arb.gui.model_output import is_reasoning_model_slug, prepare_agent_turn
from arb.gui.workflow_hints import sim_stratum
from arb.utils.io import read_jsonl

SIM_LAYER_VERSION = 4


def run_episode(
    env: GuiTaskEnvironment,
    agent,
    max_turns: int = 16,
    *,
    model_slug: str | None = None,
) -> dict:
    history: list[dict[str, str]] = []
    base_prompt = env.get_prompt()
    obs: dict = {"done": False}
    reasoning = is_reasoning_model_slug(model_slug)

    for _ in range(max_turns):
        if obs.get("done"):
            break
        if not history:
            current = base_prompt
        else:
            current = f"{base_prompt}\n\nObservation:\n{obs.get('observation', '')}"
        raw_reply = agent.complete(current, history)
        prepared = prepare_agent_turn(raw_reply, reasoning_model=reasoning)
        history.append({"role": "assistant", "content": raw_reply})
        obs = env.step(prepared.prepared)
        if not obs.get("done") and obs.get("observation"):
            history.append({"role": "system", "content": obs["observation"]})

    if not obs.get("done"):
        obs = env.finalize_timeout()

    task_dict = env.task.to_dict()
    parser_assists = getattr(env, "_parser_assists", [])
    model_native = obs.get("success", False) and not any(
        a.get("kind") == "workflow_recovery" for a in parser_assists
    )
    return {
        "task_id": env.task.id,
        "state": env.task.state,
        "success": obs.get("success", False),
        "model_native_success": model_native,
        "metrics": obs.get("metrics", {}),
        "failure_reason": obs.get("failure_reason"),
        "failure_report_valid": obs.get("failure_report_valid"),
        "failure_report_reject": obs.get("failure_report_reject"),
        "sim_stratum": sim_stratum(task_dict),
        "sim_layer_version": SIM_LAYER_VERSION,
        "parser_assists": parser_assists,
        "parser_assist_count": len(parser_assists),
        "transcript": env.transcript,
        "exploit_detect_reason": obs.get("exploit_detect_reason"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on GUI benchmark")
    parser.add_argument("--benchmark", required=True, help="Path to gui *_solvable.jsonl etc.")
    parser.add_argument("--output", default="results_gui.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--agent",
        default="honest_baseline",
        choices=["honest_baseline", "exploit_baseline", "local_hf"],
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-slug", default=None)
    parser.add_argument("--max-turns", type=int, default=16)
    args = parser.parse_args()

    if args.agent == "honest_baseline":
        agent = GuiHonestBaselineAgent()
    elif args.agent == "exploit_baseline":
        agent = GuiExploitBaselineAgent()
    else:
        if not args.model_path:
            raise ValueError("--model-path required for local_hf agent")
        agent = LocalHFAgent(args.model_path)

    model_slug = args.model_slug
    if model_slug:
        os.environ["ARB_MODEL_SLUG"] = model_slug

    results = []
    tasks = list(read_jsonl(args.benchmark))
    if args.limit:
        tasks = tasks[: args.limit]

    for task in tqdm(tasks, desc="gui_eval"):
        env = GuiTaskEnvironment(task)
        results.append(run_episode(env, agent, max_turns=args.max_turns, model_slug=model_slug))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate_results(results)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(results)} episodes -> {out_path}")


if __name__ == "__main__":
    main()

"""Code + Solvable: LiveCodeBench with unified interface."""

from __future__ import annotations

from typing import Any

from arb.schemas import CodeTaskInstance, TaskState, TaskSubstrate


def _default_filesystem_policy() -> dict[str, Any]:
    return {
        "read_only": ["tests/", ".arb/grader_config.json", ".arb/harness.py"],
        "writable": ["solution.py"],
        "clarification_enabled": False,
    }


def build_solvable_instance(base: dict[str, Any], index: int) -> CodeTaskInstance:
    split = base["split"]
    source_id = base["source_id"]
    tid = f"code_solvable_{split}_{index:05d}"

    return CodeTaskInstance(
        id=tid,
        substrate=TaskSubstrate.CODE.value,
        state=TaskState.SOLVABLE.value,
        source_dataset=base["source_dataset"],
        source_id=source_id,
        split=split,
        question_title=base["question_title"],
        question_content=base["question_content"],
        starter_code=base["starter_code"],
        platform=base["platform"],
        difficulty=base["difficulty"],
        test_type=base["test_type"],
        func_name=base.get("func_name"),
        public_tests=base["public_tests"],
        private_tests=base["private_tests"],
        clarification_enabled=False,
        clarification_channels=[],
        expected_behavior="implement_and_pass_tests",
        filesystem_policy=_default_filesystem_policy(),
        transform_ops=["livecodebench_direct"],
        metadata={
            "contest_id": base.get("contest_id"),
            "contest_date": base.get("contest_date"),
            "original_metadata": base.get("metadata", {}),
        },
    )

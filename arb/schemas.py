"""JSON schemas and datatypes for benchmark instances."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskSubstrate(str, Enum):
    TEXT = "text"
    CODE = "code"
    TOOL = "tool"
    GUI = "gui"


class TaskState(str, Enum):
    SOLVABLE = "solvable"
    BLOCKED_BUT_RECOVERABLE = "blocked_but_recoverable"
    IMPOSSIBLE_NO_EXPLOIT = "impossible_no_exploit"
    IMPOSSIBLE_WITH_EXPLOIT = "impossible_with_exploit"


@dataclass
class MaskedSlot:
    slot_id: str
    original_text: str
    original_value: str
    masked_token: str
    clarification_keywords: list[str]
    semantic_type: str  # e.g. price, count, duration, rate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExploitSignal:
    signal_type: str
    description: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextTaskInstance:
    """Unified text task record (one row in benchmark JSONL)."""

    id: str
    substrate: str
    state: str
    source_dataset: str
    source_id: str
    split: str
    question: str
    gold_answer: str | None = None
    gold_answer_numeric: float | None = None
    # Blocked
    masked_slots: list[dict[str, Any]] = field(default_factory=list)
    clarification_enabled: bool = False
    # Impossible
    impossibility_type: str | None = None  # underdetermined | conflicting
    expected_behavior: str | None = None
    # Exploit (MCQ)
    answer_format: str = "free_form"  # free_form | multiple_choice
    choices: list[str] = field(default_factory=list)
    gold_choice_index: int | None = None
    exploit_signals: list[dict[str, Any]] = field(default_factory=list)
    # Construction audit trail
    transform_ops: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TextTaskInstance:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class MaskedSpec:
    spec_id: str
    section_type: str  # constraint | io_format | edge_case | readme
    original_text: str
    masked_placeholder: str
    clarification_keywords: list[str]
    reveal_channel: str  # ask_user | open_spec | request_example

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExploitChannel:
    channel_type: str  # writable_tests | editable_grader | sample_cache | patchable_harness
    path: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodeTaskInstance:
    """Unified code task record (one row in benchmark JSONL)."""

    id: str
    substrate: str
    state: str
    source_dataset: str
    source_id: str
    split: str
    question_title: str
    question_content: str
    starter_code: str
    platform: str
    difficulty: str
    test_type: str  # functional | stdin
    func_name: str | None = None
    public_tests: list[dict[str, Any]] = field(default_factory=list)
    private_tests: list[dict[str, Any]] = field(default_factory=list)
    # Blocked
    masked_specs: list[dict[str, Any]] = field(default_factory=list)
    clarification_enabled: bool = False
    clarification_channels: list[str] = field(default_factory=list)
    # Impossible
    impossibility_type: str | None = None
    expected_behavior: str | None = None
    # Exploit
    exploit_channels: list[dict[str, Any]] = field(default_factory=list)
    filesystem_policy: dict[str, Any] = field(default_factory=dict)
    # Construction audit trail
    transform_ops: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodeTaskInstance:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class MaskedToolSlot:
    slot_id: str
    slot_type: str  # order_id | zip_code | email | confirmation
    original_text: str
    masked_token: str
    clarification_keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExploitChannel:
    channel_type: str  # debug_eval_cache | mock_confirm_api | shadow_status_log
    tool_name: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolTaskInstance:
    """Unified tool-calling task record (τ-bench derived)."""

    id: str
    substrate: str
    state: str
    source_dataset: str
    source_id: str
    split: str
    domain: str
    user_instruction: str
    domain_policy: str
    available_tools: list[str]
    gold_actions: list[dict[str, Any]]
    expected_outputs: list[str] = field(default_factory=list)
    user_id: str = ""
    # Blocked
    masked_slots: list[dict[str, Any]] = field(default_factory=list)
    clarification_enabled: bool = False
    # Impossible
    impossibility_type: str | None = None
    expected_behavior: str | None = None
    backend_overrides: dict[str, Any] = field(default_factory=dict)
    # Exploit
    exploit_channels: list[dict[str, Any]] = field(default_factory=list)
    tool_policy: dict[str, Any] = field(default_factory=dict)
    transform_ops: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolTaskInstance:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class BlockedResource:
    resource_id: str
    resource_type: str  # file | window | permission | page
    path: str
    blocked_state: str
    original_state: str
    recovery_channel: str  # request_access | ask_user | open_attachment
    clarification_keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuiExploitChannel:
    channel_type: str  # gold_artifact | success_flag | dom_stub | hidden_answer_file
    path: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuiTaskInstance:
    """Unified GUI task record (one row in benchmark JSONL)."""

    id: str
    substrate: str
    state: str
    source_dataset: str
    source_id: str
    domain: str
    split: str
    instruction: str
    snapshot: str
    related_apps: list[str]
    original_config: list[dict[str, Any]] = field(default_factory=list)
    original_evaluator: dict[str, Any] = field(default_factory=dict)
    sim_spec: dict[str, Any] = field(default_factory=dict)
    # Blocked
    blocked_resources: list[dict[str, Any]] = field(default_factory=list)
    clarification_enabled: bool = False
    recovery_channels: list[str] = field(default_factory=list)
    # Impossible
    impossibility_type: str | None = None
    expected_behavior: str | None = None
    # Exploit
    exploit_channels: list[dict[str, Any]] = field(default_factory=list)
    filesystem_policy: dict[str, Any] = field(default_factory=dict)
    # Construction audit trail
    transform_ops: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GuiTaskInstance:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AgentTurn:
    role: str  # agent | system | user
    content: str
    action: str | None = None  # answer | ask_for_clarification | report_failure
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    task_id: str
    state: str
    success: bool
    metrics: dict[str, float]
    transcript: list[dict[str, Any]]
    final_answer: str | None = None
    failure_reason: str | None = None

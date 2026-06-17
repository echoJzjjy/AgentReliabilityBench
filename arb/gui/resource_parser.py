"""Identify critical GUI resources for blocked / impossible transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CriticalResource:
    resource_id: str
    resource_type: str
    path: str
    label: str
    clarification_keywords: list[str]
    recovery_channel: str


def select_critical_resource(base: dict[str, Any], index: int) -> CriticalResource | None:
    spec = base.get("sim_spec", {})
    artifacts = spec.get("artifacts", []) or []
    apps = spec.get("apps", []) or []

    if artifacts:
        art = artifacts[index % len(artifacts)]
        basename = art.get("basename", art.get("path", "file"))
        return CriticalResource(
            resource_id=f"file_{index}",
            resource_type="file",
            path=art["path"],
            label=basename,
            clarification_keywords=[basename.lower(), "file", "attachment", "document", "open"],
            recovery_channel="open_attachment",
        )

    if apps:
        app = apps[index % len(apps)]
        name = app.get("name", "app")
        return CriticalResource(
            resource_id=f"window_{index}",
            resource_type="window",
            path=name,
            label=name,
            clarification_keywords=[name.lower(), "window", "login", "credential", "access", "open"],
            recovery_channel="request_access",
        )

    predicate = spec.get("success_predicate", {})
    if predicate.get("type") == "setting_equals":
        app = predicate.get("app", "app")
        key = predicate.get("key", "setting")
        return CriticalResource(
            resource_id=f"permission_{index}",
            resource_type="permission",
            path=f"{app}.{key}",
            label=f"{app} {key}",
            clarification_keywords=[app.lower(), key.lower(), "permission", "access", "settings"],
            recovery_channel="request_access",
        )

    return None

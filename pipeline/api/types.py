"""Typed helpers for declarative pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, List, Dict



class CombineMode(str, Enum):
    """Combination strategies for multi-pipeline runs."""

    INDEPENDENT = "independent"
    UNION = "union"
    JOIN = "join"


@dataclass
class PipelineSpecConfig:
    """Normalized configuration for a single declarative pipeline."""

    name: str
    pipeline_args: dict[str, Any]
    task_cpus: Optional[int] = None
    task_gpus: Optional[int] = None

    REQUIRED_FIELDS = {"sources", "output"}
    RESERVED_FIELDS = {
        "name",
        "task_cpus",
        "task_gpus",
    }

    @classmethod
    def from_dict(
        cls,
        raw_spec: dict[str, Any],
        defaults: Optional[Dict[str, Any]] = None,
        index: int = 0,
    ) -> PipelineSpecConfig:
        if not isinstance(raw_spec, dict):
            raise TypeError(f"Pipeline spec #{index + 1} must be a dictionary")

        merged: dict[str, Any] = {}
        if defaults:
            merged.update(defaults)
        merged.update(raw_spec)

        name = merged.pop("name", f"pipeline_{index + 1}")
        task_cpus = merged.pop("task_cpus", None)
        task_gpus = merged.pop("task_gpus", None)

        missing = cls.REQUIRED_FIELDS - merged.keys()
        if missing:
            raise ValueError(f"Pipeline '{name}' missing required fields: {missing}")

        sources = merged["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Pipeline '{name}' must define a non-empty sources list")

        return cls(
            name=name,
            pipeline_args=merged,
            task_cpus=task_cpus,
            task_gpus=task_gpus,
        )


@dataclass
class CombineConfig:
    """Configuration describing how multiple pipelines are combined."""

    mode: CombineMode = CombineMode.INDEPENDENT
    output: Optional[str] = None
    join_keys: Optional[List[str]] = None
    join_type: str = "inner"
    collect_stats: bool = False

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]] = None) -> CombineConfig:
        if not config:
            return cls()

        mode = CombineMode(config.get("mode", "independent").lower())
        join_keys = config.get("join_keys")
        if mode == CombineMode.JOIN and (not join_keys or not isinstance(join_keys, list)):
            raise ValueError("combine_config.join_keys must be provided for join mode")

        return cls(
            mode=mode,
            output=config.get("output"),
            join_keys=join_keys,
            join_type=config.get("join_type", "inner"),
            collect_stats=bool(config.get("collect_stats", False)),
        )



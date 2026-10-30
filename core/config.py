"""Validated settings; only standard library and host dependencies are used."""

import unicodedata
from dataclasses import dataclass, field

from .text import DEFAULT_ALPHABET, validate_alphabet


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    alphabet: str = DEFAULT_ALPHABET
    presets: dict[str, str] = field(
        default_factory=lambda: {"兽音": "嗷呜啊~", "喵音": "喵呜啊~"}
    )
    wait_seconds: int = 60
    max_rounds: int = 10
    max_text: int = 1000
    max_output: int = 12000
    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 8_000_000
    max_side: int = 4000
    workers: int = 2
    queue_size: int = 8
    task_seconds: int = 60
    pending_limit: int = 512

    @classmethod
    def parse(cls, data: dict) -> "Settings":
        values = {}
        for key, low, high in (
            ("wait_seconds", 10, 300),
            ("max_rounds", 1, 100),
            ("max_text", 1, 10000),
            ("max_output", 100, 100000),
            ("max_bytes", 1024, 50 * 1024 * 1024),
            ("max_pixels", 1, 8_000_000),
            ("max_side", 1, 4000),
            ("workers", 1, 4),
            ("queue_size", 0, 32),
            ("task_seconds", 5, 120),
            ("pending_limit", 1, 2048),
        ):
            value = data.get(key, getattr(cls, key))
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} 必须为 {low}–{high} 的整数")
            values[key] = value
        if type(data.get("enabled", True)) is not bool:
            raise ValueError("enabled 必须是布尔值")
        presets = {}
        entries = data.get("presets", ["兽音=嗷呜啊~", "喵音=喵呜啊~"])
        if not isinstance(entries, list) or len(entries) > 64:
            raise ValueError("预设必须为不超过 64 项的列表")
        for entry in entries:
            if not isinstance(entry, str) or "=" not in entry:
                raise ValueError("预设格式为 两字名称=四字加密词")
            name, alphabet = entry.split("=", 1)
            if (
                len(name) != 2
                or name in presets
                or any(
                    c.isspace() or unicodedata.category(c).startswith("C") for c in name
                )
            ):
                raise ValueError("预设名称须为两个可见字符，且不能重复")
            presets[name] = validate_alphabet(alphabet)
        return cls(
            enabled=data.get("enabled", True),
            alphabet=validate_alphabet(data.get("alphabet", DEFAULT_ALPHABET)),
            presets=presets,
            **values,
        )

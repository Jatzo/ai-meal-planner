"""Token limits, an estimated price, and a ledger of what each request cost."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

CHARS_PER_TOKEN = 4  # a rough rule for English text, not a tokeniser
MAX_OUTPUT_TOKENS = 300  # the most a model reply may use, and what an estimate assumes it will
PER_REQUEST_TOKEN_CAP = 1_500  # one request may not be estimated above this
DAILY_TOKEN_CAP = 50_000  # model tokens allowed per UTC day, across all requests
# Illustrative prices in US dollars, not a quote from any provider.
INPUT_USD_PER_MILLION = 3.00
OUTPUT_USD_PER_MILLION = 15.00

KINDS = ("rules", "cache", "model", "refused", "none")
# Fixed codes only, so a food name or a model's reason can never reach the log.
OUTCOMES = ("served", "rejected", "no_match", "over_request_cap", "over_daily_cap")


def estimate_tokens(prompt: str) -> int:
    """Estimate the tokens one model call could use, reply included.

    Characters divided by CHARS_PER_TOKEN is a rough rule, not a tokeniser. It errs a little
    either way, which is fine for a cap that only has to stop big mistakes.
    """
    return math.ceil(len(prompt) / CHARS_PER_TOKEN) + MAX_OUTPUT_TOKENS


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_USD_PER_MILLION + output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Entry:
    time: str  # ISO 8601, UTC
    kind: str
    fingerprint: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    outcome: str


class Ledger:
    """An append-only record of every request: how it was answered and what it cost.

    It holds fingerprints and codes only, never food names. The daily cap is
    worked out from the entries, so reloading the file keeps the cap across restarts.
    """

    def __init__(self, path: Path | None = None, clock: Callable[[], datetime] = _utc_now):
        self._path = path
        self._clock = clock
        self._entries: list[Entry] = []
        if path is not None and path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            self._entries = [Entry(**json.loads(line)) for line in lines if line.strip()]

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def record(
        self,
        kind: str,
        fingerprint: str,
        outcome: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Entry:
        if kind not in KINDS:
            raise ValueError(f"'{kind}' is not a known kind. Use one of {', '.join(KINDS)}.")
        if outcome not in OUTCOMES:
            raise ValueError(
                f"'{outcome}' is not a known outcome. Use one of {', '.join(OUTCOMES)}."
            )
        entry = Entry(
            self._clock().isoformat(),
            kind,
            fingerprint,
            input_tokens,
            output_tokens,
            cost_usd(input_tokens, output_tokens),
            outcome,
        )
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(entry)) + "\n")
        self._entries.append(entry)
        return entry

    def tokens_used_today(self) -> int:
        today = self._clock().date()
        return sum(
            entry.input_tokens + entry.output_tokens
            for entry in self._entries
            if entry.kind == "model" and datetime.fromisoformat(entry.time).date() == today
        )

    def budget_problem(self, estimated_tokens: int) -> str | None:
        """The outcome code for a call that would break a cap, or None if it fits."""
        if estimated_tokens > PER_REQUEST_TOKEN_CAP:
            return "over_request_cap"
        if self.tokens_used_today() + estimated_tokens > DAILY_TOKEN_CAP:
            return "over_daily_cap"
        return None

    def summary(self) -> str:
        made = [entry for entry in self._entries if entry.kind == "model"]
        avoided = sum(1 for entry in self._entries if entry.kind in ("rules", "cache"))
        tokens = sum(entry.input_tokens + entry.output_tokens for entry in made)
        total_cost = sum(entry.cost_usd for entry in self._entries)
        return (
            f"calls made: {len(made)}, avoided: {avoided}, tokens: {tokens} "
            f"(about ${total_cost:.4f}, illustrative)"
        )

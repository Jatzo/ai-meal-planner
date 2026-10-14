from datetime import datetime, timedelta, timezone

import pytest

from mealplanner.cost import (
    DAILY_TOKEN_CAP,
    KINDS,
    MAX_OUTPUT_TOKENS,
    OUTCOMES,
    PER_REQUEST_TOKEN_CAP,
    Ledger,
    cost_usd,
    estimate_tokens,
)
from mealplanner.models import Food, Request

TODAY = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self, now=TODAY):
        self.now = now

    def __call__(self):
        return self.now


def test_a_400_character_prompt_is_estimated_at_400_tokens():
    # 400 / 4 = 100 for the prompt, plus 300 allowed for the reply.
    assert estimate_tokens("x" * 400) == 400


def test_a_prompt_that_does_not_divide_by_four_rounds_up():
    assert estimate_tokens("x" * 401) == 101 + MAX_OUTPUT_TOKENS


def test_cost_uses_the_illustrative_prices_per_million_tokens():
    assert cost_usd(1_000_000, 0) == pytest.approx(3.00)
    assert cost_usd(0, 1_000_000) == pytest.approx(15.00)


def test_a_prompt_over_the_per_request_cap_is_refused():
    ledger = Ledger(clock=FakeClock())
    too_long = estimate_tokens("x" * 4 * PER_REQUEST_TOKEN_CAP)
    assert ledger.budget_problem(too_long) == "over_request_cap"


def test_a_prompt_within_the_caps_is_allowed():
    assert Ledger(clock=FakeClock()).budget_problem(PER_REQUEST_TOKEN_CAP) is None


def test_a_request_that_would_push_today_past_the_daily_cap_is_refused():
    ledger = Ledger(clock=FakeClock())
    ledger.record("model", "abc", "served", input_tokens=40_000, output_tokens=9_000)
    assert ledger.budget_problem(1_001) == "over_daily_cap"
    assert ledger.budget_problem(1_000) is None  # exactly at the cap is allowed


def test_yesterdays_tokens_do_not_count_towards_todays_cap():
    clock = FakeClock(TODAY - timedelta(days=1))
    ledger = Ledger(clock=clock)
    ledger.record("model", "abc", "served", input_tokens=DAILY_TOKEN_CAP, output_tokens=0)
    assert ledger.budget_problem(100) == "over_daily_cap"
    clock.now = TODAY
    assert ledger.tokens_used_today() == 0
    assert ledger.budget_problem(100) is None


def test_only_model_calls_count_towards_the_daily_cap():
    ledger = Ledger(clock=FakeClock())
    ledger.record("rules", "abc", "served", input_tokens=900, output_tokens=900)
    assert ledger.tokens_used_today() == 0


def test_the_ledger_persists_and_a_new_ledger_reloads_every_entry(tmp_path):
    path = tmp_path / ".mealplanner" / "ledger.jsonl"
    first = Ledger(path, clock=FakeClock())
    first.record("rules", "aaa", "served")
    first.record("model", "bbb", "served", input_tokens=1_000, output_tokens=200)
    first.record("refused", "ccc", "over_request_cap")

    second = Ledger(path, clock=FakeClock())
    assert second.entries == first.entries
    assert second.summary() == first.summary()
    assert second.tokens_used_today() == 1_200


def test_the_daily_cap_survives_a_restart(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path, clock=FakeClock()).record(
        "model", "abc", "served", input_tokens=DAILY_TOKEN_CAP, output_tokens=0
    )
    assert Ledger(path, clock=FakeClock()).budget_problem(1) == "over_daily_cap"


def test_new_entries_are_appended_and_earlier_lines_are_left_alone(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path, clock=FakeClock()).record("rules", "aaa", "served")
    before = path.read_text(encoding="utf-8")
    Ledger(path, clock=FakeClock()).record("cache", "bbb", "served")
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before)
    assert len(after.splitlines()) == 2


def test_the_file_holds_the_fingerprint_and_not_the_food_names(tmp_path):
    foods = {
        "peanut butter": Food("peanut butter", 588, 25, frozenset({"peanut"})),
        "salmon fillet": Food("salmon fillet", 208, 20, frozenset({"fish"})),
    }
    request = Request.from_dict(
        {"on_hand": ["peanut butter", "salmon fillet"], "kcal_target": 600, "protein_target_g": 40},
        foods,
    )
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path, clock=FakeClock())
    ledger.record("rules", request.fingerprint(), "served")
    ledger.record("model", request.fingerprint(), "rejected", input_tokens=500, output_tokens=100)

    text = path.read_text(encoding="utf-8")
    assert request.fingerprint() in text
    assert "peanut" not in text
    assert "salmon" not in text


def test_an_unknown_kind_is_refused(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", clock=FakeClock())
    with pytest.raises(ValueError, match="kind"):
        ledger.record("gossip", "abc", "served")
    assert ledger.entries == ()
    assert not (tmp_path / "ledger.jsonl").exists()


def test_an_unknown_outcome_is_refused_so_free_text_never_reaches_the_log():
    ledger = Ledger(clock=FakeClock())
    with pytest.raises(ValueError, match="outcome"):
        ledger.record("model", "abc", "the model said peanut butter was fine")


def test_every_listed_kind_and_outcome_can_be_recorded():
    ledger = Ledger(clock=FakeClock())
    for kind in KINDS:
        for outcome in OUTCOMES:
            ledger.record(kind, "abc", outcome)
    assert len(ledger.entries) == len(KINDS) * len(OUTCOMES)


def test_an_entry_records_the_time_the_cost_and_the_tokens():
    entry = Ledger(clock=FakeClock()).record(
        "model", "abc", "served", input_tokens=1_000, output_tokens=200
    )
    assert entry.time == TODAY.isoformat()
    assert entry.cost_usd == pytest.approx(0.006)
    assert (entry.input_tokens, entry.output_tokens) == (1_000, 200)


def test_the_summary_counts_calls_avoided_tokens_and_cost():
    ledger = Ledger(clock=FakeClock())
    ledger.record("rules", "a", "served")
    ledger.record("cache", "b", "served")
    ledger.record("model", "c", "served", input_tokens=1_000, output_tokens=200)
    ledger.record("refused", "d", "over_daily_cap")
    ledger.record("none", "e", "no_match")
    assert ledger.summary() == "calls made: 1, avoided: 2, tokens: 1200 (about $0.0060, illustrative)"


def test_an_empty_ledger_has_a_summary_of_zeros():
    assert Ledger(clock=FakeClock()).summary() == (
        "calls made: 0, avoided: 0, tokens: 0 (about $0.0000, illustrative)"
    )

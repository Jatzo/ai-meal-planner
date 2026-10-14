import json
import re
import sys
from pathlib import Path

import pytest

from mealplanner.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
RULES_ONLY = str(EXAMPLES / "rules_only.json")
FORCES_FALLBACK = str(EXAMPLES / "forces_fallback.json")
FOOD_NAMES = ("egg", "potato", "spinach", "tomato")


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def last_line(text):
    return text.strip().splitlines()[-1]


def write_requests(tmp_path, content):
    path = tmp_path / "requests.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return str(path)


def test_the_rules_only_example_makes_no_model_calls_and_avoids_three(capsys):
    code, out, err = run(capsys, RULES_ONLY)
    assert code == 0
    assert err == ""
    assert last_line(out) == "calls made: 0, avoided: 3, tokens: 0 (about $0.0000, illustrative)"
    assert out.count(": rules") == 3


def test_each_request_shows_its_fingerprint_its_answer_and_its_suggestions(capsys):
    _, out, _ = run(capsys, RULES_ONLY)
    assert re.search(r"^Request 1 \(fingerprint [0-9a-f]{12}\): rules$", out, re.MULTILINE)
    assert "\n  - " in out


def test_the_fallback_example_with_the_stub_makes_one_call_and_avoids_one(capsys):
    code, out, _ = run(capsys, FORCES_FALLBACK, "--model", "stub")
    assert code == 0
    assert last_line(out).startswith("calls made: 1, avoided: 1, tokens: ")
    assert ": model" in out
    assert ": cache" in out
    assert "stub meal" in out


def test_the_fallback_example_with_no_model_says_what_to_do(capsys):
    code, out, _ = run(capsys, FORCES_FALLBACK)
    assert code == 0
    assert "No recipe fits" in out
    assert "--model stub" in out
    assert last_line(out) == "calls made: 0, avoided: 0, tokens: 0 (about $0.0000, illustrative)"


def test_the_ledger_file_holds_fingerprints_and_no_food_names(capsys, tmp_path):
    path = tmp_path / ".mealplanner" / "ledger.jsonl"
    code, out, _ = run(capsys, FORCES_FALLBACK, "--model", "stub", "--ledger", str(path))
    assert code == 0
    text = path.read_text(encoding="utf-8")
    fingerprint = re.search(r"fingerprint ([0-9a-f]{12})", out).group(1)
    assert fingerprint in text
    for name in FOOD_NAMES:
        assert name not in text


def test_a_second_run_with_the_same_ledger_file_carries_the_totals_on(capsys, tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    run(capsys, RULES_ONLY, "--ledger", path)
    _, out, _ = run(capsys, RULES_ONLY, "--ledger", path)
    assert last_line(out).startswith("calls made: 0, avoided: 6, ")


def assert_plain_failure(code, err):
    assert code == 1
    assert err.strip() != ""
    assert "Traceback" not in err
    assert "try again" in err


def test_a_missing_file_returns_1_with_a_plain_message(capsys, tmp_path):
    code, out, err = run(capsys, str(tmp_path / "nothing.json"))
    assert_plain_failure(code, err)
    assert out == ""


def test_a_file_that_is_not_json_returns_1_with_a_plain_message(capsys, tmp_path):
    code, _, err = run(capsys, write_requests(tmp_path, "{not json"))
    assert_plain_failure(code, err)


def test_a_file_that_is_not_a_list_returns_1_with_a_plain_message(capsys, tmp_path):
    code, _, err = run(capsys, write_requests(tmp_path, {"on_hand": ["egg"]}))
    assert_plain_failure(code, err)


def test_a_bad_request_returns_1_and_says_which_request_it_was(capsys, tmp_path):
    good = {"on_hand": ["egg"], "kcal_target": 500, "protein_target_g": 30}
    bad = {"on_hand": [], "kcal_target": 500, "protein_target_g": 30}
    code, out, err = run(capsys, write_requests(tmp_path, [good, bad]))
    assert_plain_failure(code, err)
    assert "Request 2" in err
    assert out == ""  # nothing is answered until every request has been checked


def test_bedrock_without_a_model_id_returns_1_and_asks_for_one(capsys):
    code, _, err = run(capsys, RULES_ONLY, "--model", "bedrock")
    assert_plain_failure(code, err)
    assert "--bedrock-model-id" in err


def test_bedrock_with_boto3_missing_returns_1_and_names_the_bedrock_extra(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    code, _, err = run(capsys, RULES_ONLY, "--model", "bedrock", "--bedrock-model-id", "some-id")
    assert_plain_failure(code, err)
    assert "bedrock extra" in err


def test_a_model_call_that_raises_returns_1_with_the_failure_message(capsys, monkeypatch):
    class FailingModel:
        def complete(self, system, prompt, request):
            raise RuntimeError("connection reset")

    monkeypatch.setattr("mealplanner.cli.StubModel", FailingModel)
    code, _, err = run(capsys, FORCES_FALLBACK, "--model", "stub")
    assert code == 1
    assert "The model call for request 1 failed" in err
    assert "connection reset" in err
    assert "Traceback" not in err


def test_a_damaged_ledger_file_returns_1_with_a_plain_message(capsys, tmp_path):
    ledger = tmp_path / ".mealplanner" / "ledger.jsonl"
    ledger.parent.mkdir()
    ledger.write_text('{"time": "2026-03-10T12:00:00+00:00", "kind": "ru', encoding="utf-8")
    code, _, err = run(capsys, RULES_ONLY, "--ledger", str(ledger))
    assert code == 1
    assert "Fix or move the file" in err
    assert "Traceback" not in err

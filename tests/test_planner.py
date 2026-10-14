import json
from datetime import datetime, timezone

import pytest

from mealplanner.cost import KINDS, Ledger
from mealplanner.llm import ModelReply, StubModel
from mealplanner.models import Request, load_foods, load_recipes
from mealplanner.planner import Planner
from mealplanner.rules import evaluate, match_recipes

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
FOODS = load_foods()
RECIPES = load_recipes()

# Four foods that no recipe can be built from, so only the model can answer.
NOTHING_FITS = frozenset({"egg", "potato", "spinach", "tomato"})


class RaisingModel:
    """Fails the test if it is ever asked for anything."""

    def complete(self, system, prompt, request):
        raise AssertionError("the model must not be called here")


def make_request(on_hand, kcal=600, protein=40, allergies=()):
    return Request(frozenset(on_hand), float(kcal), float(protein), frozenset(allergies))


def make_planner(model=None, ledger=None):
    ledger = ledger or Ledger(clock=lambda: NOW)
    return Planner(FOODS, RECIPES, ledger, model), ledger


def reply_text(*ingredients, **extra):
    meal = {"name": "test meal", "ingredients": [{"food": f, "grams": g} for f, g in ingredients]}
    return json.dumps({**meal, **extra})


def kinds(ledger):
    return [entry.kind for entry in ledger.entries]


def outcomes(ledger):
    return [entry.outcome for entry in ledger.entries]


def recipe_request():
    return make_request({i.food for recipe in RECIPES for i in recipe.ingredients}, protein=45)


def test_the_test_requests_really_match_no_recipe():
    assert match_recipes(RECIPES, FOODS, make_request(NOTHING_FITS)) == []
    assert match_recipes(RECIPES, FOODS, make_request({"peanut butter", "egg"})) == []


def test_a_recipe_that_fits_is_served_without_touching_the_model():
    planner, ledger = make_planner(RaisingModel())
    answer = planner.plan(recipe_request())
    assert answer.kind == "rules"
    assert 1 <= len(answer.suggestions) <= 3
    assert kinds(ledger) == ["rules"]
    assert outcomes(ledger) == ["served"]


def test_no_match_and_no_model_says_what_to_do():
    planner, ledger = make_planner()
    answer = planner.plan(make_request(NOTHING_FITS))
    assert answer.kind == "none"
    assert answer.suggestions == ()
    assert "--model stub" in answer.message and "--model bedrock" in answer.message
    assert kinds(ledger) == ["none"]
    assert outcomes(ledger) == ["no_match"]


def test_the_fallback_serves_a_checked_meal_with_macros_worked_out_from_the_table():
    request = make_request(NOTHING_FITS)
    planner, ledger = make_planner(StubModel())
    answer = planner.plan(request)
    assert answer.kind == "model"
    assert len(answer.suggestions) == 1
    suggestion = answer.suggestions[0]
    assert suggestion.name == "stub meal"
    # 100 g each of egg, potato, spinach and tomato, added up from foods.json.
    assert suggestion.kcal == pytest.approx(143 + 77 + 23 + 18)
    assert suggestion.protein_g == pytest.approx(13 + 2 + 2.9 + 0.9)
    assert outcomes(ledger) == ["served"]
    assert ledger.entries[0].input_tokens > 0


def test_a_peanut_dish_is_rejected_for_someone_with_a_peanut_allergy():
    request = make_request({"peanut butter", "egg"}, allergies={"peanut"})
    model = StubModel(reply_text(("peanut butter", 30), ("egg", 100)))
    planner, ledger = make_planner(model)
    answer = planner.plan(request)
    assert answer.kind == "model"
    assert answer.suggestions == ()
    assert "peanut" in answer.message
    assert outcomes(ledger) == ["rejected"]


def test_a_food_that_is_not_on_hand_is_rejected():
    model = StubModel(reply_text(("salmon fillet", 150)))
    planner, ledger = make_planner(model)
    answer = planner.plan(make_request(NOTHING_FITS))
    assert answer.suggestions == ()
    assert "salmon fillet" in answer.message
    assert outcomes(ledger) == ["rejected"]


def test_a_reply_that_is_not_json_is_rejected():
    planner, ledger = make_planner(StubModel("Sorry, I have no idea."))
    answer = planner.plan(make_request(NOTHING_FITS))
    assert answer.kind == "model"
    assert answer.suggestions == ()
    assert answer.message.startswith("The model suggested a meal, but it was not used:")
    assert outcomes(ledger) == ["rejected"]


def test_a_repeated_request_calls_the_model_once_and_the_second_answer_comes_from_the_cache():
    stub = StubModel()
    planner, ledger = make_planner(stub)
    request = make_request(NOTHING_FITS)
    first = planner.plan(request)
    second = planner.plan(request)
    assert stub.calls == 1
    assert kinds(ledger) == ["model", "cache"]
    assert outcomes(ledger) == ["served", "served"]
    assert second.kind == "cache"
    assert second.suggestions == first.suggestions
    cached_entry = ledger.entries[1]
    assert (cached_entry.input_tokens, cached_entry.output_tokens, cached_entry.cost_usd) == (0, 0, 0)


def test_a_rejected_reply_is_cached_too_so_a_repeat_never_pays_twice():
    stub = StubModel("Sorry, I have no idea.")
    planner, ledger = make_planner(stub)
    request = make_request(NOTHING_FITS)
    planner.plan(request)
    again = planner.plan(request)
    assert stub.calls == 1
    assert kinds(ledger) == ["model", "cache"]
    assert outcomes(ledger) == ["rejected", "rejected"]
    assert again.suggestions == ()


def test_a_request_over_the_daily_cap_is_refused_before_any_call_and_nothing_is_cached():
    ledger = Ledger(clock=lambda: NOW)
    ledger.record("model", "earlier", "served", input_tokens=49_900)
    planner, _ = make_planner(RaisingModel(), ledger)
    request = make_request(NOTHING_FITS)
    answer = planner.plan(request)
    assert answer.kind == "refused"
    assert answer.suggestions == ()
    assert "not asked" in answer.message
    assert ledger.entries[-1].outcome == "over_daily_cap"
    # Had the refusal cached anything, the repeat would be a "cache" entry.
    planner.plan(request)
    assert kinds(ledger) == ["model", "refused", "refused"]


def test_a_request_over_the_per_request_cap_is_refused_before_any_call(monkeypatch):
    monkeypatch.setattr("mealplanner.cost.PER_REQUEST_TOKEN_CAP", 100)
    planner, ledger = make_planner(RaisingModel())
    answer = planner.plan(make_request(NOTHING_FITS))
    assert answer.kind == "refused"
    assert outcomes(ledger) == ["over_request_cap"]
    assert ledger.entries[0].input_tokens == 0


def test_numbers_the_model_sends_are_ignored_and_the_table_decides():
    request = make_request(NOTHING_FITS)
    text = reply_text(("egg", 100), ("potato", 200), kcal=9999, protein_g=999, score=100)
    answer = make_planner(StubModel(text))[0].plan(request)
    expected = evaluate(
        "test meal",
        answer.suggestions[0].ingredients,
        FOODS,
        request,
    )
    suggestion = answer.suggestions[0]
    assert suggestion == expected
    assert suggestion.kcal == pytest.approx(143 + 154)
    assert suggestion.score < 100


def test_a_model_meal_far_from_the_target_is_still_served_when_it_passes_the_rules():
    request = make_request(NOTHING_FITS, kcal=2500, protein=150)
    answer = make_planner(StubModel())[0].plan(request)
    assert answer.suggestions[0].score < 70


def test_the_ledger_file_for_a_fallback_run_holds_the_fingerprint_and_no_food_names(tmp_path):
    path = tmp_path / ".mealplanner" / "ledger.jsonl"
    request = make_request(NOTHING_FITS)
    planner, _ = make_planner(StubModel(), Ledger(path, clock=lambda: NOW))
    planner.plan(request)
    planner.plan(request)
    text = path.read_text(encoding="utf-8")
    assert request.fingerprint() in text
    for name in NOTHING_FITS:
        assert name not in text
    assert "stub meal" not in text


def test_every_answer_kind_is_one_the_ledger_knows():
    planner, _ = make_planner(StubModel())
    for request in (recipe_request(), make_request(NOTHING_FITS)):
        assert planner.plan(request).kind in KINDS


def test_an_empty_reply_is_rejected_and_the_tokens_it_cost_are_still_recorded():
    class EmptyModel:
        def complete(self, system, prompt, request):
            return ModelReply("", 10, 0)

    planner, ledger = make_planner(EmptyModel())
    answer = planner.plan(make_request(NOTHING_FITS))
    assert answer.suggestions == ()
    assert ledger.entries[0].input_tokens == 10
    assert outcomes(ledger) == ["rejected"]


def test_a_model_that_fails_raises_a_model_call_error_and_writes_no_ledger_entry():
    from mealplanner.llm import ModelCallError

    class FailingModel:
        def complete(self, system, prompt, request):
            raise RuntimeError("connection reset")

    planner, ledger = make_planner(FailingModel())
    with pytest.raises(ModelCallError, match="connection reset"):
        planner.plan(make_request(NOTHING_FITS))
    assert ledger.entries == ()

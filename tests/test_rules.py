import math

import pytest

from mealplanner.models import Food, Ingredient, Recipe, Request, load_foods, load_recipes
from mealplanner.rules import (
    SERVE_THRESHOLD,
    evaluate,
    find_problem,
    macros,
    match_recipes,
    score,
)

# One invented food where 600 g gives exactly 660 kcal and 48 g protein.
SIMPLE_FOODS = {"fuel": Food("fuel", 110, 8, frozenset())}


def request_for(on_hand, kcal=600, protein=40, allergies=()):
    return Request(frozenset(on_hand), float(kcal), float(protein), frozenset(allergies))


def fuel(grams):
    return (Ingredient("fuel", grams),)


@pytest.fixture(scope="module")
def foods():
    return load_foods()


@pytest.fixture(scope="module")
def recipes():
    return load_recipes()


@pytest.fixture(scope="module")
def satay(recipes):
    return next(recipe for recipe in recipes if "satay" in recipe.name)


def foods_used(recipe):
    return {ingredient.food for ingredient in recipe.ingredients}


def test_an_allergen_on_the_allergy_list_rules_the_satay_recipe_out(foods, recipes, satay):
    request = request_for(
        {i.food for recipe in recipes for i in recipe.ingredients},
        allergies={"peanut"},
    )
    problem = find_problem(satay.ingredients, foods, request)
    assert "peanut butter" in problem
    assert "peanut" in problem
    assert satay.name not in [s.name for s in match_recipes(recipes, foods, request)]


def test_the_satay_recipe_has_no_problem_when_peanut_is_not_an_allergy(foods, satay):
    assert find_problem(satay.ingredients, foods, request_for(foods_used(satay))) is None


def test_a_food_that_is_not_on_hand_is_a_problem(foods):
    problem = find_problem(
        (Ingredient("chicken breast", 100), Ingredient("potato", 100)),
        foods,
        request_for({"chicken breast"}),
    )
    assert "potato" in problem


def test_an_empty_meal_is_a_problem(foods):
    assert find_problem((), foods, request_for({"potato"})) is not None


@pytest.mark.parametrize("grams", [0, -50, math.inf, math.nan])
def test_grams_that_are_zero_negative_or_not_finite_are_a_problem(grams, foods):
    problem = find_problem((Ingredient("potato", grams),), foods, request_for({"potato"}))
    assert "potato" in problem


def test_a_food_missing_from_the_table_is_a_problem(foods):
    problem = find_problem(
        (Ingredient("dragon fruit", 100),), foods, request_for({"dragon fruit"})
    )
    assert "dragon fruit" in problem


def test_a_meal_with_nothing_wrong_has_no_problem(foods):
    meal = (Ingredient("chicken breast", 130), Ingredient("potato", 200))
    assert find_problem(meal, foods, request_for({"chicken breast", "potato"})) is None


def test_the_first_problem_in_the_order_of_checks_is_the_one_reported(foods):
    # Zero grams comes before an unknown food, so the unknown food is not named.
    meal = (Ingredient("potato", 0), Ingredient("dragon fruit", 100))
    problem = find_problem(meal, foods, request_for({"potato", "dragon fruit"}))
    assert "potato" in problem
    assert "dragon fruit" not in problem


def test_macros_come_from_the_table_per_100_grams():
    assert macros(fuel(600), SIMPLE_FOODS) == (pytest.approx(660), pytest.approx(48))


def test_660_kcal_against_600_and_48_g_against_40_scores_exactly_86():
    assert score(660, 48, request_for({"fuel"})) == 86.0


def test_a_meal_on_target_scores_100():
    assert score(600, 40, request_for({"fuel"})) == 100.0


def test_the_score_never_goes_below_zero_for_a_wildly_wrong_meal():
    assert score(60000, 4000, request_for({"fuel"})) == 0.0


def test_the_reason_quotes_the_numbers_the_score_used():
    suggestion = evaluate("worked", fuel(600), SIMPLE_FOODS, request_for({"fuel"}))
    assert suggestion.score == 86.0
    assert suggestion.reason == (
        "660 kcal against a 600 kcal target (10.0% off) and "
        "48.0 g protein against 40.0 g (20.0% off), so it scores 86.0."
    )
    for number in (
        f"{suggestion.kcal:.0f}",
        f"{suggestion.protein_g:.1f}",
        f"{suggestion.score:.1f}",
        "10.0%",
        "20.0%",
    ):
        assert number in suggestion.reason


def test_a_suggestion_keeps_its_name_and_ingredients():
    suggestion = evaluate("worked", fuel(600), SIMPLE_FOODS, request_for({"fuel"}))
    assert suggestion.name == "worked"
    assert suggestion.ingredients == fuel(600)


def everything_on_hand(recipes, **targets):
    return request_for({i.food for recipe in recipes for i in recipe.ingredients}, **targets)


def test_match_recipes_returns_at_most_three_best_first_all_at_the_threshold_or_above(
    foods, recipes
):
    matches = match_recipes(recipes, foods, everything_on_hand(recipes, protein=45))
    assert len(matches) == 3
    assert all(match.score >= SERVE_THRESHOLD for match in matches)
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)


def test_match_recipes_honours_a_different_limit(foods, recipes):
    matches = match_recipes(recipes, foods, everything_on_hand(recipes, protein=45), limit=5)
    assert len(matches) == 5


def test_match_recipes_returns_nothing_when_nothing_on_hand_fits(foods, recipes):
    assert match_recipes(recipes, foods, request_for({"banana"})) == []


def test_match_recipes_returns_nothing_when_every_meal_is_too_far_from_the_target(foods, recipes):
    assert match_recipes(recipes, foods, everything_on_hand(recipes, kcal=5000)) == []


def test_match_recipes_breaks_a_tie_by_name():
    tied = [Recipe("b meal", fuel(600)), Recipe("a meal", fuel(600))]
    matches = match_recipes(tied, SIMPLE_FOODS, request_for({"fuel"}))
    assert [m.name for m in matches] == ["a meal", "b meal"]

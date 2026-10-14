import pytest

from mealplanner.models import (
    ALLERGENS,
    Request,
    RequestError,
    load_foods,
    load_recipes,
)


@pytest.fixture(scope="module")
def foods():
    return load_foods()


def good_request(**changes):
    data = {
        "on_hand": ["chicken breast", "cooked brown rice"],
        "kcal_target": 600,
        "protein_target_g": 40,
        "allergies": ["peanut"],
    }
    data.update(changes)
    return data


def test_every_recipe_food_is_in_the_food_table(foods):
    for recipe in load_recipes():
        for ingredient in recipe.ingredients:
            assert ingredient.food in foods, f"{recipe.name} uses {ingredient.food}"


def test_every_allergen_in_the_table_is_in_the_allowed_list(foods):
    for food in foods.values():
        assert food.allergens <= ALLERGENS, food.name


def test_the_table_has_at_least_one_food_for_each_allergen(foods):
    listed = set().union(*(food.allergens for food in foods.values()))
    assert listed == ALLERGENS


def test_there_are_twelve_recipes():
    assert len(load_recipes()) == 12


def test_exactly_one_recipe_contains_peanut(foods):
    peanut_recipes = [
        recipe.name
        for recipe in load_recipes()
        if any("peanut" in foods[i.food].allergens for i in recipe.ingredients)
    ]
    assert peanut_recipes == ["chicken satay with rice"]


def test_a_valid_request_is_cleaned_and_kept(foods):
    request = Request.from_dict(
        good_request(on_hand=["  Chicken Breast "], allergies=["PEANUT"]), foods
    )
    assert request.on_hand == frozenset({"chicken breast"})
    assert request.allergies == frozenset({"peanut"})
    assert request.kcal_target == 600.0


def test_allergies_may_be_left_out(foods):
    data = good_request()
    del data["allergies"]
    assert Request.from_dict(data, foods).allergies == frozenset()


@pytest.mark.parametrize(
    "data",
    [
        "not a dict",
        {"kcal_target": 600, "protein_target_g": 40},
        good_request(on_hand="chicken breast"),
        good_request(on_hand=[]),
        good_request(on_hand=["chicken breast", 7]),
        good_request(kcal_target=0),
        good_request(kcal_target=-5),
        good_request(kcal_target="600"),
        good_request(kcal_target=True),
        good_request(kcal_target=float("inf")),
        good_request(protein_target_g=0),
        good_request(protein_target_g=False),
        good_request(allergies="peanut"),
        good_request(allergies=[3]),
    ],
)
def test_a_bad_request_is_refused_with_what_to_do(data, foods):
    with pytest.raises(RequestError) as caught:
        Request.from_dict(data, foods)
    assert "try again" in str(caught.value)


def test_a_missing_key_is_named(foods):
    data = good_request()
    del data["protein_target_g"]
    with pytest.raises(RequestError, match="protein_target_g"):
        Request.from_dict(data, foods)


def test_an_unknown_food_is_named_and_the_table_is_pointed_to(foods):
    with pytest.raises(RequestError, match="dragon fruit") as caught:
        Request.from_dict(good_request(on_hand=["dragon fruit"]), foods)
    assert "foods.json" in str(caught.value)


def test_an_unknown_allergy_is_named_and_the_allowed_ones_listed(foods):
    with pytest.raises(RequestError, match="pollen") as caught:
        Request.from_dict(good_request(allergies=["pollen"]), foods)
    for allergen in ALLERGENS:
        assert allergen in str(caught.value)


def test_a_request_error_can_be_caught_as_a_value_error(foods):
    with pytest.raises(ValueError):
        Request.from_dict(good_request(on_hand=[]), foods)


def test_the_fingerprint_is_twelve_hex_characters(foods):
    fingerprint = Request.from_dict(good_request(), foods).fingerprint()
    assert len(fingerprint) == 12
    int(fingerprint, 16)


def test_the_fingerprint_ignores_the_order_of_lists_and_int_versus_float(foods):
    first = Request.from_dict(
        good_request(on_hand=["chicken breast", "potato"], allergies=["egg", "soy"]),
        foods,
    )
    second = Request.from_dict(
        good_request(
            on_hand=["potato", "chicken breast"],
            allergies=["soy", "egg"],
            kcal_target=600.0,
        ),
        foods,
    )
    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize(
    "changes",
    [
        {"kcal_target": 601},
        {"protein_target_g": 41},
        {"on_hand": ["chicken breast"]},
        {"allergies": []},
    ],
)
def test_the_fingerprint_changes_when_the_request_changes(changes, foods):
    base = Request.from_dict(good_request(), foods).fingerprint()
    assert Request.from_dict(good_request(**changes), foods).fingerprint() != base


def test_a_target_too_large_for_a_float_is_refused_with_what_to_do(foods):
    with pytest.raises(RequestError, match="then try again"):
        Request.from_dict(good_request(kcal_target=10**400), foods)


def test_a_food_table_with_an_unknown_allergen_is_refused(monkeypatch):
    row = {"name": "mystery", "kcal": 100, "protein_g": 1.0, "allergens": ["sesame"]}
    monkeypatch.setattr("mealplanner.models._read_json", lambda filename: [row])
    with pytest.raises(ValueError, match="sesame"):
        load_foods()

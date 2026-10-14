"""The rules that decide whether a meal is allowed and how well it fits the request."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from mealplanner.models import Food, Ingredient, Recipe, Request

# Below this score a meal is too far from the targets to be worth serving.
SERVE_THRESHOLD = 70


@dataclass(frozen=True)
class Suggestion:
    name: str
    ingredients: tuple[Ingredient, ...]
    kcal: float
    protein_g: float
    score: float
    reason: str


def find_problem(
    ingredients: Sequence[Ingredient], foods: dict[str, Food], request: Request
) -> str | None:
    """Return the first reason a meal cannot be served, or None if it can.

    This is the only place these checks live. Recipes and anything a model
    suggests both go through it, so neither can skip a rule.
    """
    if not ingredients:
        return "The meal has no ingredients."
    for item in ingredients:
        if not math.isfinite(item.grams) or item.grams <= 0:
            return f"{item.food} has {item.grams:g} g, and every ingredient needs more than 0 g."
    for item in ingredients:
        if item.food not in foods:
            return f"{item.food} is not in the food table."
    for item in ingredients:
        if item.food not in request.on_hand:
            return f"{item.food} is not on hand."
    for item in ingredients:
        found = foods[item.food].allergens & request.allergies
        if found:
            return f"{item.food} contains {', '.join(sorted(found))}, which is on the allergy list."
    return None


def macros(ingredients: Sequence[Ingredient], foods: dict[str, Food]) -> tuple[float, float]:
    """Total kcal and protein in grams, from the food table only.

    Call find_problem first: a food missing from the table raises KeyError here.
    """
    kcal = sum(foods[item.food].kcal * item.grams / 100 for item in ingredients)
    protein_g = sum(foods[item.food].protein_g * item.grams / 100 for item in ingredients)
    return kcal, protein_g


def _percent_off(actual: float, target: float) -> float:
    return abs(actual - target) / target * 100


def score(kcal: float, protein_g: float, request: Request) -> float:
    """100 for a meal on target, less 0.6 per kcal percent off and 0.4 per protein percent off.

    Calories carry 60% of the weight and protein 40%. Rounded to one place so
    float noise never shows.
    """
    kcal_off = _percent_off(kcal, request.kcal_target)
    protein_off = _percent_off(protein_g, request.protein_target_g)
    return round(max(0.0, 100 - 0.6 * kcal_off - 0.4 * protein_off), 1)


def evaluate(
    name: str, ingredients: Sequence[Ingredient], foods: dict[str, Food], request: Request
) -> Suggestion:
    """Work out a meal's macros and score, and say why in one sentence."""
    kcal, protein_g = macros(ingredients, foods)
    meal_score = score(kcal, protein_g, request)
    reason = (
        f"{kcal:.0f} kcal against a {request.kcal_target:g} kcal target "
        f"({_percent_off(kcal, request.kcal_target):.1f}% off) and "
        f"{protein_g:.1f} g protein against {request.protein_target_g:.1f} g "
        f"({_percent_off(protein_g, request.protein_target_g):.1f}% off), "
        f"so it scores {meal_score:.1f}."
    )
    return Suggestion(name, tuple(ingredients), kcal, protein_g, meal_score, reason)


def match_recipes(
    recipes: Sequence[Recipe], foods: dict[str, Food], request: Request, limit: int = 3
) -> list[Suggestion]:
    """The best allowed recipes at or above SERVE_THRESHOLD, best first, ties by name."""
    suggestions = [
        evaluate(recipe.name, recipe.ingredients, foods, request)
        for recipe in recipes
        if find_problem(recipe.ingredients, foods, request) is None
    ]
    suggestions = [s for s in suggestions if s.score >= SERVE_THRESHOLD]
    suggestions.sort(key=lambda s: (-s.score, s.name))
    return suggestions[:limit]

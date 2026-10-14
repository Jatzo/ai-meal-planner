"""Foods, recipes and the meal request, with the checks that keep bad input out."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources

# The only allergens the food table and a request may name.
ALLERGENS = frozenset({"peanut", "gluten", "dairy", "egg", "fish", "shellfish", "soy"})

_REQUIRED_KEYS = ("on_hand", "kcal_target", "protein_target_g")


class RequestError(ValueError):
    """The request cannot be used. The message says what happened and what to do."""


@dataclass(frozen=True)
class Food:
    name: str
    kcal: float  # per 100 g
    protein_g: float  # per 100 g
    allergens: frozenset[str]


@dataclass(frozen=True)
class Ingredient:
    food: str
    grams: float


@dataclass(frozen=True)
class Recipe:
    name: str
    ingredients: tuple[Ingredient, ...]


@dataclass(frozen=True)
class Request:
    on_hand: frozenset[str]
    kcal_target: float
    protein_target_g: float
    allergies: frozenset[str]

    @classmethod
    def from_dict(cls, data: dict, foods: dict[str, Food]) -> Request:
        """Check raw input once, at the edge, so the rest of the code can trust it."""
        if not isinstance(data, dict):
            raise RequestError(
                "The request is not a set of named fields. "
                "Send an object with on_hand, kcal_target and protein_target_g, then try again."
            )
        for key in _REQUIRED_KEYS:
            if key not in data:
                raise RequestError(f"The request has no {key}. Add {key}, then try again.")

        on_hand = _clean_names(data["on_hand"], "on_hand")
        if not on_hand:
            raise RequestError(
                "The request has no foods on hand. "
                "Add at least one food name to on_hand, then try again."
            )
        for name in sorted(on_hand):
            if name not in foods:
                raise RequestError(
                    f"The food '{name}' is not in the food table. "
                    "Use a name from foods.json, then try again."
                )

        allergies = _clean_names(data.get("allergies", []), "allergies")
        unknown = sorted(allergies - ALLERGENS)
        if unknown:
            raise RequestError(
                f"The allergy '{unknown[0]}' is not one we know. "
                f"Use only {', '.join(sorted(ALLERGENS))}, then try again."
            )

        return cls(
            on_hand=on_hand,
            kcal_target=_positive_number(data["kcal_target"], "kcal_target"),
            protein_target_g=_positive_number(data["protein_target_g"], "protein_target_g"),
            allergies=allergies,
        )

    def fingerprint(self) -> str:
        """A short, stable id for the request, so the same ask can be recognised later.

        It holds food names and numbers only. Lists are sorted so order never matters.
        """
        canonical = json.dumps(
            {
                "on_hand": sorted(self.on_hand),
                "kcal_target": self.kcal_target,
                "protein_target_g": self.protein_target_g,
                "allergies": sorted(self.allergies),
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _clean_names(value: object, key: str) -> frozenset[str]:
    """Turn a list of names into a set, stripped and lowercased, or refuse it."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RequestError(f"{key} is not a list of text names. Send a list of names, then try again.")
    return frozenset(item.strip().lower() for item in value)


def _positive_number(value: object, key: str) -> float:
    number = None
    # bool is a subclass of int, so True would otherwise pass as 1.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:  # an integer too large for a float is refused below
            pass
    if number is None or not math.isfinite(number) or number <= 0:
        raise RequestError(f"{key} must be a number above zero. Send a number above zero, then try again.")
    return number


def _read_json(filename: str) -> list[dict]:
    text = resources.files("mealplanner").joinpath("data", filename).read_text(encoding="utf-8")
    return json.loads(text)


def load_foods() -> dict[str, Food]:
    """Read the food table, refusing any allergen outside ALLERGENS."""
    foods = {}
    for row in _read_json("foods.json"):
        allergens = frozenset(row["allergens"])
        unknown = allergens - ALLERGENS
        if unknown:
            raise ValueError(
                f"foods.json lists {sorted(unknown)} for {row['name']}, "
                f"which is not an allowed allergen. Use only {sorted(ALLERGENS)}."
            )
        foods[row["name"]] = Food(row["name"], row["kcal"], row["protein_g"], allergens)
    return foods


def load_recipes() -> list[Recipe]:
    return [
        Recipe(
            row["name"],
            tuple(Ingredient(item["food"], item["grams"]) for item in row["ingredients"]),
        )
        for row in _read_json("recipes.json")
    ]

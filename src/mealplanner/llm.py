"""The language model behind a small interface, and a strict reader for what it says.

The model is a fallback that only proposes a meal. Whatever it returns is read
here down to a name and ingredients, then checked by the same rules as a recipe.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Protocol

from mealplanner.cost import CHARS_PER_TOKEN, MAX_OUTPUT_TOKENS
from mealplanner.models import Food, Ingredient, Request

SYSTEM_PROMPT = (
    "You suggest one single-meal recipe. Reply with JSON only, in this shape: "
    '{"name": str, "ingredients": [{"food": str, "grams": number}]}. '
    "Use only the foods listed, and avoid the listed allergens."
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ModelReplyError(ValueError):
    """The model's reply cannot be used. The message gives the plain reason."""


class ModelUnavailableError(RuntimeError):
    """The model cannot be reached or set up on this machine."""


class ModelCallError(RuntimeError):
    """The call to the model itself failed, so there is no reply to check."""


@dataclass(frozen=True)
class ModelReply:
    text: str
    input_tokens: int
    output_tokens: int


class Model(Protocol):
    def complete(self, system: str, prompt: str, request: Request) -> ModelReply: ...


def build_prompt(request: Request, foods: dict[str, Food]) -> str:
    """Describe the ask with food names, numbers and allergen names only.

    Nothing else about the person goes in, and it stays short enough for the
    per-request token cap.
    """
    lines = ["Foods on hand, per 100 g:"]
    for name in sorted(request.on_hand):
        food = foods[name]
        lines.append(f"- {name}: {food.kcal:g} kcal, {food.protein_g:g} g protein")
    allergies = ", ".join(sorted(request.allergies)) or "none"
    lines.append(f"Target: {request.kcal_target:g} kcal, {request.protein_target_g:g} g protein.")
    lines.append(f"Allergies: {allergies}")
    return "\n".join(lines)


def parse_reply(text: str) -> tuple[str, tuple[Ingredient, ...]]:
    """Read a reply down to a name and ingredients, and drop everything else.

    A model may add its own kcal, protein or score. Those are never trusted:
    the numbers are worked out again from the food table.
    """
    stripped = text.strip()
    fenced = _FENCE.fullmatch(stripped)
    try:
        data = json.loads(fenced.group(1) if fenced else stripped)
    except json.JSONDecodeError as error:
        raise ModelReplyError("The model reply is not valid JSON, so it was not used.") from error
    if not isinstance(data, dict):
        raise ModelReplyError("The model reply is not a JSON object, so it was not used.")

    name = data.get("name")
    if not isinstance(name, str):
        raise ModelReplyError("The model reply has no meal name as text, so it was not used.")
    items = data.get("ingredients")
    if not isinstance(items, list):
        raise ModelReplyError("The model reply has no list of ingredients, so it was not used.")

    ingredients = []
    for item in items:
        if not isinstance(item, dict):
            raise ModelReplyError("An ingredient in the model reply is not an object, so it was not used.")
        food, grams = item.get("food"), item.get("grams")
        if not isinstance(food, str):
            raise ModelReplyError("An ingredient in the model reply has no food name as text, so it was not used.")
        # bool is a subclass of int, so True would otherwise pass as 1 g.
        if isinstance(grams, bool) or not isinstance(grams, (int, float)):
            raise ModelReplyError(f"The grams for {food.strip()} are not a number, so the reply was not used.")
        try:
            grams = float(grams)
        except OverflowError:  # an integer too large for a float
            raise ModelReplyError(
                f"The grams for {food.strip()} are too large, so the reply was not used."
            ) from None
        ingredients.append(Ingredient(food.strip().lower(), grams))
    return name, tuple(ingredients)


class StubModel:
    """A model that makes no network call, for tests and for running offline.

    With no reply given it proposes the first four on-hand foods at 100 g each.
    With a reply given it returns that text every time.
    """

    def __init__(self, reply: str | None = None):
        self._reply = reply
        self.calls = 0

    def complete(self, system: str, prompt: str, request: Request) -> ModelReply:
        self.calls += 1
        text = self._reply
        if text is None:
            foods = sorted(request.on_hand)[:4]
            text = json.dumps(
                {"name": "stub meal", "ingredients": [{"food": food, "grams": 100} for food in foods]}
            )
        return ModelReply(
            text,
            math.ceil(len(system + prompt) / CHARS_PER_TOKEN),
            math.ceil(len(text) / CHARS_PER_TOKEN),
        )


class BedrockModel:
    """Amazon Bedrock through its Converse call. The request object is not sent."""

    def __init__(self, model_id: str, region: str | None = None, client=None):
        if client is None:
            try:
                import boto3
            except ImportError as error:
                raise ModelUnavailableError(
                    "boto3 is not installed, so the Bedrock model cannot run. "
                    "Install the bedrock extra with pip install -e \".[bedrock]\", then try again."
                ) from error
            client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
        self._client = client

    def complete(self, system: str, prompt: str, request: Request) -> ModelReply:
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": MAX_OUTPUT_TOKENS, "temperature": 0},
        )
        try:
            return ModelReply(
                response["output"]["message"]["content"][0]["text"],
                response["usage"]["inputTokens"],
                response["usage"]["outputTokens"],
            )
        except (KeyError, IndexError, TypeError) as error:
            raise ModelReplyError(
                "The Bedrock response did not have the expected text and token counts, so it was not used."
            ) from error

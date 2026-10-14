"""Answers a request: rules first, then a remembered reply, then a checked model call."""

from __future__ import annotations

from dataclasses import dataclass

from mealplanner.cost import Ledger, estimate_tokens
from mealplanner.llm import (
    SYSTEM_PROMPT,
    Model,
    ModelCallError,
    ModelReplyError,
    build_prompt,
    parse_reply,
)
from mealplanner.models import Food, Recipe, Request
from mealplanner.rules import Suggestion, evaluate, find_problem, match_recipes


@dataclass(frozen=True)
class Answer:
    kind: str  # one of cost.KINDS
    suggestions: tuple[Suggestion, ...]
    message: str  # for a person; may name foods, so it is never logged


_REFUSALS = {
    "over_request_cap": (
        "The model was not asked, because this request is estimated at {estimated} tokens, "
        "over the cap for one request. Use fewer foods in on_hand, then try again."
    ),
    "over_daily_cap": (
        "The model was not asked, because it would pass today's token cap. "
        "Try again tomorrow, or change the foods or targets so a recipe fits."
    ),
}


class Planner:
    def __init__(
        self,
        foods: dict[str, Food],
        recipes: list[Recipe],
        ledger: Ledger,
        model: Model | None = None,
    ):
        self._foods = foods
        self._recipes = recipes
        self._ledger = ledger
        self._model = model
        self.cache: dict[str, str] = {}  # fingerprint -> raw model reply text

    def plan(self, request: Request) -> Answer:
        """Answer in a fixed order and record exactly one ledger entry.

        1. Recipes that fit. The model is never touched.
        2. No fit and no model: say so.
        3. A reply kept from an earlier identical request, checked again.
        4. A budget check before any model call.
        5. One model call. Its meal is served when it passes find_problem alone,
           with no score threshold, and its score and reason are shown so the
           person can judge how close it is. The model's own numbers are ignored.
        """
        fingerprint = request.fingerprint()

        matches = match_recipes(self._recipes, self._foods, request)
        if matches:
            self._ledger.record("rules", fingerprint, "served")
            fits = "1 recipe fits" if len(matches) == 1 else f"{len(matches)} recipes fit"
            return Answer("rules", tuple(matches), f"{fits} what you have on hand.")

        if self._model is None:
            self._ledger.record("none", fingerprint, "no_match")
            return Answer(
                "none",
                (),
                "No recipe fits the foods you have on hand and your targets. "
                "Run with --model stub or --model bedrock to ask for a suggestion.",
            )

        if fingerprint in self.cache:
            suggestion, reason = self._validate(self.cache[fingerprint], request)
            self._ledger.record("cache", fingerprint, "served" if suggestion else "rejected")
            return self._answer("cache", suggestion, reason)

        prompt = build_prompt(request, self._foods)
        estimated = estimate_tokens(SYSTEM_PROMPT + prompt)
        problem = self._ledger.budget_problem(estimated)
        if problem:
            self._ledger.record("refused", fingerprint, problem)
            return Answer("refused", (), _REFUSALS[problem].format(estimated=estimated))

        try:
            reply = self._model.complete(SYSTEM_PROMPT, prompt, request)
        except Exception as error:  # network, credentials or anything else inside the model
            # No reply means no usage figures to count, so no ledger entry is written.
            raise ModelCallError(str(error)) from error
        # Kept even if rejected below, so asking again never pays twice.
        self.cache[fingerprint] = reply.text
        suggestion, reason = self._validate(reply.text, request)
        self._ledger.record(
            "model",
            fingerprint,
            "served" if suggestion else "rejected",
            reply.input_tokens,
            reply.output_tokens,
        )
        return self._answer("model", suggestion, reason)

    def _validate(self, text: str, request: Request) -> tuple[Suggestion | None, str]:
        """Read and check a model reply. Fresh and remembered replies both come through here."""
        try:
            name, ingredients = parse_reply(text)
        except ModelReplyError as error:
            return None, str(error)
        problem = find_problem(ingredients, self._foods, request)
        if problem:
            return None, problem
        suggestion = evaluate(name, ingredients, self._foods, request)
        return suggestion, suggestion.reason

    @staticmethod
    def _answer(kind: str, suggestion: Suggestion | None, reason: str) -> Answer:
        if suggestion is None:
            return Answer(
                kind,
                (),
                f"The model suggested a meal, but it was not used: {reason} "
                "Change the foods on hand or the targets, then try again.",
            )
        return Answer(
            kind,
            (suggestion,),
            "No recipe fits, so the model suggested a meal. It passed every rule, "
            "so it is served whatever its score.",
        )

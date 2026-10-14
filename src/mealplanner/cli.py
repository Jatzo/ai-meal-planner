"""Command line: read a file of requests, answer each one, then print what it cost."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mealplanner.cost import Ledger
from mealplanner.llm import BedrockModel, ModelCallError, ModelUnavailableError, StubModel
from mealplanner.models import Request, RequestError, load_foods, load_recipes
from mealplanner.planner import Planner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mealplanner",
        description="Suggest a meal from the foods you have on hand. Recipes come first.",
    )
    parser.add_argument("file", help="a JSON file holding a list of requests")
    parser.add_argument(
        "--model",
        choices=["none", "stub", "bedrock"],
        default="none",
        help="who to ask when no recipe fits (default: none)",
    )
    parser.add_argument("--bedrock-model-id", help="the Bedrock model to ask, needed with --model bedrock")
    parser.add_argument("--region", help="the AWS region for Bedrock")
    parser.add_argument(
        "--ledger",
        help="keep the cost ledger in this file, for example .mealplanner/ledger.jsonl "
        "(default: in memory only)",
    )
    return parser


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _load_requests(path: str, foods: dict) -> list[Request] | str:
    """Read every request before any is answered, so a typo never costs a model call.

    Returns the requests, or a plain sentence saying what is wrong.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return f"The file {path} could not be read. Check the path, then try again."
    except json.JSONDecodeError:
        return f"The file {path} is not valid JSON. Fix it, then try again."
    if not isinstance(data, list):
        return f"The file {path} does not hold a list of requests. Wrap the request in [ ], then try again."
    requests = []
    for number, item in enumerate(data, start=1):
        try:
            requests.append(Request.from_dict(item, foods))
        except RequestError as error:
            return f"Request {number}: {error}"
    return requests


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model == "bedrock" and not args.bedrock_model_id:
        return _fail("--model bedrock needs a model. Add --bedrock-model-id, then try again.")

    foods = load_foods()
    requests = _load_requests(args.file, foods)
    if isinstance(requests, str):
        return _fail(requests)

    try:
        model = {
            "none": lambda: None,
            "stub": StubModel,
            "bedrock": lambda: BedrockModel(args.bedrock_model_id, args.region),
        }[args.model]()
    except ModelUnavailableError as error:
        return _fail(str(error))

    try:
        ledger = Ledger(Path(args.ledger) if args.ledger else None)
    except (OSError, ValueError, TypeError):
        return _fail(
            f"The ledger file {args.ledger} could not be read, or has a line that is not a ledger "
            "entry. Fix or move the file, then try again."
        )
    planner = Planner(foods, load_recipes(), ledger, model)

    for number, request in enumerate(requests, start=1):
        try:
            answer = planner.plan(request)
        except ModelCallError as error:
            # A failed call writes no ledger entry, so nothing is counted for it.
            return _fail(
                f"The model call for request {number} failed, so nothing was served for it: "
                f"{error}. Check the model settings (for Bedrock, your AWS credentials and "
                "region), then try again."
            )
        print(f"Request {number} (fingerprint {request.fingerprint()}): {answer.kind}")
        print(f"  {answer.message}")
        for suggestion in answer.suggestions:
            print(f"  - {suggestion.name}: {suggestion.reason}")
            print("    " + ", ".join(f"{i.food} {i.grams:g} g" for i in suggestion.ingredients))
        print()
    print(ledger.summary())
    return 0

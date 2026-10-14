import json
import sys

import pytest

from mealplanner.cost import MAX_OUTPUT_TOKENS, PER_REQUEST_TOKEN_CAP, estimate_tokens
from mealplanner.llm import (
    SYSTEM_PROMPT,
    BedrockModel,
    ModelReply,
    ModelReplyError,
    ModelUnavailableError,
    StubModel,
    build_prompt,
    parse_reply,
)
from mealplanner.models import Ingredient, Request, load_foods

GOOD_REPLY = '{"name": "Salmon Plate", "ingredients": [{"food": " Salmon Fillet ", "grams": 150}]}'


def request_for(on_hand, kcal=600, protein=40, allergies=()):
    return Request(frozenset(on_hand), float(kcal), float(protein), frozenset(allergies))


class FakeBedrockClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


CANNED_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": GOOD_REPLY}]}},
    "usage": {"inputTokens": 321, "outputTokens": 45, "totalTokens": 366},
}


def test_the_converse_request_has_exactly_the_agreed_shape():
    client = FakeBedrockClient(CANNED_RESPONSE)
    BedrockModel("some-model-id", client=client).complete("the system", "the prompt", request_for({"egg"}))
    assert client.calls == [
        {
            "modelId": "some-model-id",
            "system": [{"text": "the system"}],
            "messages": [{"role": "user", "content": [{"text": "the prompt"}]}],
            "inferenceConfig": {"maxTokens": 300, "temperature": 0},
        }
    ]
    assert MAX_OUTPUT_TOKENS == 300


def test_the_reply_text_and_token_counts_are_read_from_the_response():
    model = BedrockModel("some-model-id", client=FakeBedrockClient(CANNED_RESPONSE))
    reply = model.complete("s", "p", request_for({"egg"}))
    assert reply == ModelReply(text=GOOD_REPLY, input_tokens=321, output_tokens=45)


@pytest.mark.parametrize(
    "response",
    [{}, {"output": {"message": {"content": []}}, "usage": {}}, {"output": "oops"}, None],
)
def test_a_response_without_the_expected_shape_is_refused(response):
    model = BedrockModel("some-model-id", client=FakeBedrockClient(response))
    with pytest.raises(ModelReplyError):
        model.complete("s", "p", request_for({"egg"}))


def test_a_missing_boto3_gives_a_plain_message_naming_the_bedrock_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(ModelUnavailableError) as caught:
        BedrockModel("some-model-id")
    assert str(caught.value) == (
        "boto3 is not installed, so the Bedrock model cannot run. "
        "Install the bedrock extra with pip install -e \".[bedrock]\", then try again."
    )


def test_plain_json_is_parsed_with_names_cleaned():
    name, ingredients = parse_reply(GOOD_REPLY)
    assert name == "Salmon Plate"
    assert ingredients == (Ingredient("salmon fillet", 150.0),)


@pytest.mark.parametrize("fence", ["```json\n{}\n```", "```\n{}\n```", "  ```json\n{}```  "])
def test_json_wrapped_in_a_code_fence_is_still_parsed(fence):
    assert parse_reply(fence.replace("{}", GOOD_REPLY))[0] == "Salmon Plate"


@pytest.mark.parametrize(
    "text",
    [
        "I would suggest a salmon plate.",
        "[1, 2]",
        '{"ingredients": []}',
        '{"name": 7, "ingredients": []}',
        '{"name": "x", "ingredients": "salmon"}',
        '{"name": "x"}',
        '{"name": "x", "ingredients": ["salmon"]}',
        '{"name": "x", "ingredients": [{"food": 4, "grams": 100}]}',
        '{"name": "x", "ingredients": [{"grams": 100}]}',
        '{"name": "x", "ingredients": [{"food": "egg", "grams": "100"}]}',
        '{"name": "x", "ingredients": [{"food": "egg", "grams": true}]}',
        '{"name": "x", "ingredients": [{"food": "egg"}]}',
    ],
)
def test_a_reply_that_is_not_usable_raises_a_model_reply_error(text):
    with pytest.raises(ModelReplyError):
        parse_reply(text)


def test_a_model_reply_error_can_be_caught_as_a_value_error():
    with pytest.raises(ValueError):
        parse_reply("not json")


def test_numbers_the_model_adds_are_dropped_and_only_name_food_and_grams_survive():
    text = json.dumps(
        {
            "name": "Plate",
            "kcal": 1,
            "score": 100,
            "ingredients": [{"food": "egg", "grams": 100, "kcal": 1, "protein_g": 99}],
        }
    )
    assert parse_reply(text) == ("Plate", (Ingredient("egg", 100.0),))


def test_the_stub_default_reply_uses_the_first_four_on_hand_foods_alphabetically_at_100_g():
    request = request_for({"salmon fillet", "egg", "spinach", "tomato", "potato"})
    reply = StubModel().complete("s", "p", request)
    assert parse_reply(reply.text) == (
        "stub meal",
        tuple(Ingredient(food, 100.0) for food in ("egg", "potato", "salmon fillet", "spinach")),
    )


def test_the_stub_default_reply_is_the_same_on_every_call():
    request = request_for({"egg", "potato"})
    stub = StubModel()
    assert stub.complete("s", "p", request) == stub.complete("s", "p", request)


def test_the_stub_uses_fewer_foods_when_fewer_are_on_hand():
    _, ingredients = parse_reply(StubModel().complete("s", "p", request_for({"egg"})).text)
    assert ingredients == (Ingredient("egg", 100.0),)


def test_the_stub_returns_a_fixed_reply_verbatim_and_counts_its_calls():
    stub = StubModel("not json at all")
    assert stub.calls == 0
    first = stub.complete("system", "prompt", request_for({"egg"}))
    second = stub.complete("system", "prompt", request_for({"egg"}))
    assert first.text == second.text == "not json at all"
    assert stub.calls == 2


def test_the_stub_counts_tokens_as_characters_over_four_rounded_up():
    reply = StubModel("x" * 9).complete("a" * 4, "b" * 5, request_for({"egg"}))
    assert (reply.input_tokens, reply.output_tokens) == (3, 3)


def test_the_prompt_lists_sorted_foods_with_numbers_targets_and_allergies():
    foods = load_foods()
    prompt = build_prompt(request_for({"salmon fillet", "egg"}, 600, 40, {"peanut", "dairy"}), foods)
    assert prompt.index("egg: 143 kcal, 13 g protein") < prompt.index("salmon fillet: 208 kcal, 20 g protein")
    assert "600 kcal" in prompt
    assert "40 g protein" in prompt
    assert "dairy, peanut" in prompt


def test_the_prompt_says_none_when_there_are_no_allergies():
    prompt = build_prompt(request_for({"egg"}), load_foods())
    assert "Allergies: none" in prompt


def test_the_system_prompt_names_the_reply_shape():
    assert "JSON only" in SYSTEM_PROMPT
    assert '"ingredients"' in SYSTEM_PROMPT and '"grams"' in SYSTEM_PROMPT


@pytest.mark.parametrize("count", [10, 31])
def test_the_prompt_fits_under_the_per_request_cap(count):
    foods = load_foods()
    request = request_for(sorted(foods)[:count], allergies={"peanut", "gluten"})
    estimate = estimate_tokens(SYSTEM_PROMPT + build_prompt(request, foods))
    assert estimate <= PER_REQUEST_TOKEN_CAP


def test_grams_too_large_for_a_float_are_refused_with_a_plain_reason():
    with pytest.raises(ModelReplyError, match="too large"):
        parse_reply('{"name": "x", "ingredients": [{"food": "egg", "grams": ' + "9" * 400 + "}]}")

import http.client
import json
import threading

import pytest

from mealplanner.models import ALLERGENS, load_foods
from mealplanner.web import MAX_BODY_BYTES, make_server

RECIPE_FITS = {
    "on_hand": ["chicken breast", "cooked brown rice", "broccoli", "olive oil"],
    "kcal_target": 580,
    "protein_target_g": 48,
}
# The same values as examples/forces_fallback.json: no recipe fits, and the stub meal passes every rule.
NONE_FIT = {"on_hand": ["egg", "potato", "spinach", "tomato"], "kcal_target": 600, "protein_target_g": 40}


@pytest.fixture
def server():
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def call(server, method, path, body=None, content_type="application/json", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    sent_headers = {"Content-Type": content_type, **(headers or {})}
    if isinstance(body, dict):
        body = json.dumps(body)
    connection.request(method, path, body=body, headers=sent_headers)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data


def plan(server, body):
    status, data = call(server, "POST", "/api/plan", body)
    return status, json.loads(data)


def test_the_home_page_is_served_from_the_package(server):
    status, data = call(server, "GET", "/")
    assert status == 200
    assert b"<title>Rules-first meal planner</title>" in data


def test_a_query_string_does_not_hide_the_home_page(server):
    status, _ = call(server, "GET", "/?from=link")
    assert status == 200


def test_the_food_list_holds_every_food_and_allergen(server):
    status, data = call(server, "GET", "/api/foods")
    body = json.loads(data)
    assert status == 200
    assert sorted(body["foods"]) == sorted(load_foods())
    assert body["allergens"] == sorted(ALLERGENS)


def test_a_request_a_recipe_fits_is_answered_by_the_rules(server):
    status, body = plan(server, {**RECIPE_FITS, "model": "none"})
    assert status == 200
    assert body["kind"] == "rules"
    assert body["suggestions"][0]["name"] == "chicken and rice bowl"
    assert body["summary"].startswith("calls made: 0, avoided: 1")


def test_with_no_model_and_no_fit_the_page_does_not_mention_command_flags(server):
    status, body = plan(server, {**NONE_FIT, "model": "none"})
    assert status == 200
    assert body["kind"] == "none"
    assert "--model" not in body["message"]


def test_the_stub_model_is_asked_once_and_the_repeat_comes_from_the_cache(server):
    _, first = plan(server, {**NONE_FIT, "model": "stub"})
    _, second = plan(server, {**NONE_FIT, "on_hand": list(reversed(NONE_FIT["on_hand"])), "model": "stub"})
    assert first["kind"] == "model"
    assert second["kind"] == "cache"
    assert len(second["suggestions"]) == 1
    assert second["summary"].startswith("calls made: 1, avoided: 1")


def test_a_food_outside_the_table_is_refused_with_the_reason(server):
    status, body = plan(server, {**RECIPE_FITS, "on_hand": ["dragon fruit"]})
    assert status == 400
    assert "dragon fruit" in body["error"]


@pytest.mark.parametrize("model", ["bedrock", [], {"name": "stub"}])
def test_an_unknown_model_is_refused(server, model):
    status, body = plan(server, {**RECIPE_FITS, "model": model})
    assert status == 400
    assert "stub model" in body["error"]


@pytest.mark.parametrize("body", ["not json", "[" * 5000, "[1, 2]", b"\xff\xfe"])
def test_a_body_that_is_not_a_json_object_is_refused(server, body):
    status, data = call(server, "POST", "/api/plan", body)
    assert status == 400
    assert "error" in json.loads(data)


def test_a_body_over_the_size_cap_is_refused_before_it_is_read(server):
    status, _ = call(server, "POST", "/api/plan", "x" * (MAX_BODY_BYTES + 1))
    assert status == 413


def test_a_post_that_is_not_json_is_refused_so_other_sites_cannot_post_a_form(server):
    status, _ = call(server, "POST", "/api/plan", "on_hand=egg", content_type="application/x-www-form-urlencoded")
    assert status == 415


def test_an_unknown_address_is_not_found(server):
    assert call(server, "GET", "/admin")[0] == 404
    assert call(server, "POST", "/api/other", {})[0] == 404


def test_a_second_server_cannot_take_a_port_already_in_use(server):
    with pytest.raises(OSError):
        make_server(port=server.server_port)

"""A small local web page for trying the planner in a browser.

Standard library only, so the package keeps its promise of no runtime dependencies.
It listens on 127.0.0.1 alone: the page is for showing the planner on your own screen.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import urlsplit

from mealplanner.cost import Ledger
from mealplanner.llm import ModelCallError, StubModel
from mealplanner.models import ALLERGENS, Request, RequestError, load_foods, load_recipes
from mealplanner.planner import Answer, Planner

# A real request is a few hundred bytes. Anything far bigger is refused before it is read.
MAX_BODY_BYTES = 10_000

# The planner's own wording names a command-line flag, which means nothing on the page.
_NO_MATCH_ON_THE_PAGE = (
    "No recipe fits the foods you have on hand and your targets. "
    "Switch on the stub model to ask it for a suggestion."
)


class PlannerService:
    """Holds one planner per model choice, sharing one ledger, so counts carry across clicks."""

    def __init__(self) -> None:
        self.foods = load_foods()
        recipes = load_recipes()
        self.ledger = Ledger()
        self._planners = {
            "none": Planner(self.foods, recipes, self.ledger, None),
            "stub": Planner(self.foods, recipes, self.ledger, StubModel()),
        }
        # The server answers on several threads; the planner and ledger expect one at a time.
        self._lock = threading.Lock()

    def plan(self, body: dict) -> tuple[HTTPStatus, dict]:
        model = body.get("model", "none")
        # A list or object here cannot be looked up in a dict, so it is refused before the lookup.
        if not isinstance(model, str) or model not in self._planners:
            return HTTPStatus.BAD_REQUEST, {"error": "Choose recipes only or the stub model, then try again."}
        fields = {key: value for key, value in body.items() if key != "model"}
        try:
            request = Request.from_dict(fields, self.foods)
        except RequestError as error:
            return HTTPStatus.BAD_REQUEST, {"error": str(error)}
        with self._lock:
            try:
                answer = self._planners[model].plan(request)
            except ModelCallError as error:
                return HTTPStatus.BAD_GATEWAY, {"error": f"The model call failed, so nothing was served: {error}"}
            summary = self.ledger.summary()
        return HTTPStatus.OK, _answer_json(answer, request.fingerprint(), summary)


def _answer_json(answer: Answer, fingerprint: str, summary: str) -> dict:
    message = _NO_MATCH_ON_THE_PAGE if answer.kind == "none" else answer.message
    return {
        "kind": answer.kind,
        "message": message,
        "fingerprint": fingerprint,
        "suggestions": [
            {
                "name": suggestion.name,
                "kcal": round(suggestion.kcal),
                "protein_g": round(suggestion.protein_g, 1),
                "score": suggestion.score,
                "reason": suggestion.reason,
                "ingredients": [{"food": item.food, "grams": item.grams} for item in suggestion.ingredients],
            }
            for suggestion in answer.suggestions
        ],
        "summary": summary,
    }


def _make_handler(service: PlannerService, page: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif path == "/api/foods":
                self._send_json(
                    HTTPStatus.OK,
                    {"foods": sorted(service.foods), "allergens": sorted(ALLERGENS)},
                )
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "There is nothing at this address."})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/plan":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "There is nothing at this address."})
                return
            # Requiring JSON stops another site in the same browser from posting a plain form here.
            content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type != "application/json":
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send the request as JSON, then try again."}
                )
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "The request has no length. Send it again."})
                return
            if length < 0:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "The request length is not valid. Send it again."})
                return
            if length > MAX_BODY_BYTES:
                # The body is left unread, so the connection closes rather than being reused.
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "The request is too large. Send fewer foods, then try again."},
                    close=True,
                )
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, RecursionError):  # bad JSON, bad UTF-8, or nesting deep enough to overflow
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "The request is not valid JSON. Send it again."})
                return
            if not isinstance(body, dict):
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"error": "The request is not a set of named fields. Send it again."}
                )
                return
            status, result = service.plan(body)
            self._send_json(status, result)

        def _send_json(self, status: HTTPStatus, data: dict, close: bool = False) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json", close)

        def _send(self, status: HTTPStatus, body: bytes, content_type: str, close: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            # The default prints every request, which clutters the terminal during a demo.
            pass

    return Handler


class LocalServer(ThreadingHTTPServer):
    # On Windows the default lets a second server bind a port already in use, and fail silently.
    allow_reuse_address = False
    daemon_threads = True


def make_server(port: int = 8000) -> LocalServer:
    page = resources.files("mealplanner").joinpath("static", "index.html").read_bytes()
    return LocalServer(("127.0.0.1", port), _make_handler(PlannerService(), page))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mealplanner.web",
        description="Open the meal planner in your browser, on this computer only.",
    )
    parser.add_argument("--port", type=int, default=8000, help="the port to listen on (default: 8000)")
    args = parser.parse_args(argv)
    try:
        server = make_server(args.port)
    except OSError:
        print(
            f"Port {args.port} is already in use. Close whatever is using it, "
            "or add --port 8001, then try again.",
            file=sys.stderr,
        )
        return 1
    print(f"The planner is open at http://127.0.0.1:{server.server_port}/ . Press Ctrl+C to stop it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

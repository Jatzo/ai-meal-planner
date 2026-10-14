# Rules-first meal planner

A small Python programme that suggests a meal from the foods you have on hand. A rules engine answers first, using 12 recipes and a scoring formula. A language model is asked only when no recipe fits, and only if you have switched one on. Every model call is counted and capped, and everything the model sends back goes through the same rule checks as a recipe. Its own numbers are thrown away and worked out again from the food table.

The data is invented: 31 foods and 12 recipes, written for this project. This is not dietary or medical advice.

The reasoning behind the design is in [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

From the project root, in PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[test]"
python -m pytest
python -m mealplanner examples/rules_only.json
python -m mealplanner examples/forces_fallback.json --model stub
```

On macOS or Linux, use `python3.13 -m venv .venv` and `source .venv/bin/activate`.

The package needs Python 3.11 or later and has no runtime dependencies. The tests also run in GitHub Actions on every push.

## Two example runs

The first file holds three ordinary requests.

```powershell
python -m mealplanner examples/rules_only.json
```

```text
Request 1 (fingerprint e7c8b84daa16): rules
  1 recipe fits what you have on hand.
  - chicken and rice bowl: 583 kcal against a 580 kcal target (0.5% off) and 48.5 g protein against 48.0 g (1.0% off), so it scores 99.3.
    chicken breast 130 g, cooked brown rice 200 g, broccoli 100 g, olive oil 10 g

Request 2 (fingerprint dbf0740c1878): rules
  1 recipe fits what you have on hand.
  - salmon with sweet potato and spinach: 569 kcal against a 570 kcal target (0.2% off) and 34.3 g protein against 34.0 g (0.9% off), so it scores 99.5.
    salmon fillet 140 g, sweet potato 250 g, spinach 80 g, olive oil 5 g

Request 3 (fingerprint b2b0abef75ae): rules
  1 recipe fits what you have on hand.
  - yoghurt and oat bowl: 559 kcal against a 560 kcal target (0.2% off) and 31.4 g protein against 31.0 g (1.3% off), so it scores 99.4.
    rolled oats 60 g, greek yoghurt 250 g, banana 100 g

calls made: 0, avoided: 3, tokens: 0 (about $0.0000, illustrative)
```

Rules answered all three requests. No model was called.

The second file asks twice for the same thing, with the foods listed in a different order. No recipe fits, so the stub model (a stand-in that makes no network call) is asked.

```powershell
python -m mealplanner examples/forces_fallback.json --model stub
```

```text
Request 1 (fingerprint e6576bf2270a): model
  No recipe fits, so the model suggested a meal. It passed every rule, so it is served whatever its score.
  - stub meal: 261 kcal against a 600 kcal target (56.5% off) and 18.8 g protein against 40.0 g (53.0% off), so it scores 44.9.
    egg 100 g, potato 100 g, spinach 100 g, tomato 100 g

Request 2 (fingerprint e6576bf2270a): cache
  No recipe fits, so the model suggested a meal. It passed every rule, so it is served whatever its score.
  - stub meal: 261 kcal against a 600 kcal target (56.5% off) and 18.8 g protein against 40.0 g (53.0% off), so it scores 44.9.
    egg 100 g, potato 100 g, spinach 100 g, tomato 100 g

calls made: 1, avoided: 1, tokens: 142 (about $0.0009, illustrative)
```

The repeat came from the cache, so one model call was made and one was avoided. The meal scores 44.9 and is still shown, because it passed every rule. The score is there so you can judge it.

Run without a model, the same file gets a plain "No recipe fits" for both requests and no calls at all.

## Request file format

The file is a JSON list. Each item is one request.

```json
[
  {
    "on_hand": ["salmon fillet", "sweet potato", "spinach", "olive oil"],
    "kcal_target": 570,
    "protein_target_g": 34,
    "allergies": ["dairy"]
  }
]
```

- `on_hand`: a list of food names, at least one, each taken from the food table (`src/mealplanner/data/foods.json`). Names are trimmed and lowercased.
- `kcal_target`: calories wanted for one meal, a number above zero.
- `protein_target_g`: grams of protein wanted for one meal, a number above zero.
- `allergies`: optional. Any of `peanut`, `gluten`, `dairy`, `egg`, `fish`, `shellfish`, `soy`.

The whole file is checked before any request is answered, so a typo in request 3 never costs a model call for request 1. The error says which request and what to change.

## Optional ledger

By default the cost record lives in memory for one run. To keep it in a file:

```powershell
python -m mealplanner examples/forces_fallback.json --model stub --ledger .mealplanner/ledger.jsonl
```

Each decision adds one JSON line holding the time, the kind (`rules`, `cache`, `model`, `refused` or `none`), the request fingerprint, input and output tokens, the illustrative cost and an outcome code. It holds no food names. The `.mealplanner/` folder is gitignored.

A kept ledger also keeps the daily token cap across runs. With the default in-memory ledger, the daily count starts again at zero each time.

## Running against Amazon Bedrock

This is for you to run yourself. It has not been run against AWS here.

```powershell
pip install -e ".[bedrock]"
python -m mealplanner examples/forces_fallback.json --model bedrock --bedrock-model-id amazon.nova-lite-v1:0 --region us-east-1
```

You need AWS credentials in the usual places and model access switched on for that model in that region. Any model that supports the Converse API can be named instead. Without boto3 installed, the command stops and tells you to install the bedrock extra.

## Honest limits

- The Bedrock path has been tested only against a fake client. No real AWS call has been made.
- The data is invented. Nothing here is dietary or medical advice.
- Characters divided by 4 is a rough token estimate.
- The prices (3 and 15 US dollars per million input and output tokens) are illustrative, not a quote.
- The cache lives in memory for one run. A new run starts with an empty cache.
- A meal from the model is served on the rules alone, whatever its score. It shows its score so you can judge it.
- A failed model call writes no ledger entry, so nothing is counted for it.

## Layout

- `src/mealplanner/models.py`: foods, recipes, the request, and the checks on input.
- `src/mealplanner/rules.py`: the allow checks, the macros, the score and the recipe matcher.
- `src/mealplanner/cost.py`: token estimate, the two caps, illustrative prices and the ledger.
- `src/mealplanner/llm.py`: the model interface, the strict reply reader, the stub and the Bedrock model.
- `src/mealplanner/planner.py`: the fixed order of rules, cache, budget and model.
- `src/mealplanner/cli.py`: reads a request file, prints each answer and the summary line.
- `src/mealplanner/data/`: the invented food table and recipes.
- `examples/`: the two request files above.
- `tests/`: the pytest suite.
- `.github/workflows/ci.yml`: runs the tests and both examples.

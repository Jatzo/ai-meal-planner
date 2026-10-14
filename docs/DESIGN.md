# Design notes

The numbers here all come from the code, and the data is invented. Nothing here is dietary or medical advice.

## When not to use a model

Most requests do not need one. A request names some foods, a calorie target and a protein target. Twelve recipes and one formula can answer that, the answer is the same every time, and it costs nothing.

So the model is asked only when no recipe fits, and only if you have configured one. In the first example run, rules answered all three requests and the summary line read `calls made: 0, avoided: 3`. A model would have cost money and could have been wrong.

## The planner order

`Planner.plan` works through five steps in a fixed order and writes exactly one ledger entry for each request.

1. **Rules.** Score every allowed recipe. If any scores 70 or more, serve up to 3 and stop. The model is never touched.
2. **None.** If nothing fits and no model is configured, say so.
3. **Cache.** If the same request (by fingerprint) was already sent to the model in this run, reuse its stored reply and check it again.
4. **Budget.** Estimate the tokens for the call. If it would break either cap, refuse and do not call.
5. **Model.** Make one call, keep the raw reply in the cache, then check it.

Recipes and model output go through one gate, `find_problem`, so neither can skip a rule. It runs five checks in order and returns the first problem it finds:

1. The meal has ingredients.
2. Every ingredient has a finite amount above 0 g.
3. Every food is in the food table.
4. Every food is on hand.
5. No food contains an allergen on the allergy list.

If the model's reply is not valid JSON, or has the wrong shape, it is rejected before these checks run. A rejected reply is still cached, so asking again never pays twice.

## The score

```text
score = max(0, 100 - 0.6 * kcal_percent_off - 0.4 * protein_percent_off)
```

Calories carry 60% of the weight and protein 40%. The result is rounded to one decimal place. A recipe must score at least 70 to be served.

A worked example: a meal of 660 kcal and 48 g protein, against a target of 600 kcal and 40 g.

- Calories are 10% off, which costs 0.6 x 10 = 6 points.
- Protein is 20% off, which costs 0.4 x 20 = 8 points.
- The score is 100 - 6 - 8 = 86.0.

The reason string the code produces for that meal is:

```text
660 kcal against a 600 kcal target (10.0% off) and 48.0 g protein against 40.0 g (20.0% off), so it scores 86.0.
```

The same sentence is printed for recipes and for model meals, so you read both the same way.

## The cost model

Before a model call, the planner estimates its size as the prompt length in characters divided by 4 (rounded up), plus 300 for the reply. It is a rough estimate, close enough to stop big mistakes.

Two caps apply:

- **1,500 tokens per request.** Over that, the call is refused. With the current 31-food table this cap is unlikely to be reached; it is there for a larger table.
- **50,000 tokens per UTC day.** The count comes from the model entries in the ledger, so it survives a restart only if you keep the ledger in a file with `--ledger`.

Prices are illustrative: 3 US dollars per million input tokens and 15 per million output tokens. They are not a quote from any provider.

The run ends with one summary line:

```text
calls made: 1, avoided: 1, tokens: 142 (about $0.0009, illustrative)
```

How the counts work:

- **Made:** entries of kind `model`.
- **Avoided:** entries of kind `rules` plus kind `cache`.
- **Neither:** `refused` (a cap stopped the call) and `none` (no recipe fit and no model was configured).

Tokens and cost come from the model's own usage figures. For the stub, they are estimated from text length.

## Governance

**An append-only decision log.** With `--ledger`, each decision adds one JSON line and nothing is ever rewritten. A line holds the time, the kind, a request fingerprint, the token counts, the illustrative cost and an outcome code. The outcome codes are fixed: `served`, `rejected`, `no_match`, `over_request_cap` and `over_daily_cap`. The ledger refuses any other value, so a food name or a model's own words cannot reach the log. The fingerprint is the first 12 characters of a SHA-256 hash of the sorted request. It is not encryption: anyone with the food table could guess a request and check it.

**Allergies are hard constraints.** An allergen on the list rules a food out in `find_problem`, before any score is worked out. A good score never outweighs an allergy, for recipes or for model meals.

**No personal data goes to a model.** The prompt holds food names, calories and protein per 100 g, the two targets, and allergen names. It holds no name, address, account, date, free text or fingerprint, and the request object itself is not sent. The allergy list is health-related, though, so keep that in mind before you point this at a real service.

**The model's numbers are discarded.** The reader keeps only a meal name and a list of foods with grams. If the model adds its own calories, protein or score, they are dropped. The planner works them out again from the food table.

**A person should review any model-suggested meal before relying on it.** Passing the rules means it uses only foods on hand and avoids the listed allergens. It can still be a poor meal. The stub meal in the second example scores 44.9 and is still served, with its score showing.

## Limits

- The Bedrock path has been tested only against a fake client. It has not been run against AWS.
- The data is invented, and this is not dietary or medical advice.
- The in-memory cache lasts for one run, and the in-memory daily count restarts with it.
- The planner could be exposed as a single MCP tool: one call, a request in and an answer out. That is not built here.

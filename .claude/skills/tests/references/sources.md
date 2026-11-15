# Sources

## Published

| Rule | Source |
|---|---|
| See a test fail before trusting it | Seemann, why trust tests: https://blog.ploeh.dk/2013/04/02/why-trust-tests/; mutation testing is the formal version |
| Test via public APIs; state, not interactions | *Software Engineering at Google* ch. 12 and 13: https://abseil.io/resources/swe-book/html/ch12.html, https://abseil.io/resources/swe-book/html/ch13.html |
| One behaviour per test, named after it | SWE book ch. 12 |
| No logic in tests | SWE book ch. 12, "Don't Put Logic in Tests" |
| DAMP, not DRY | SWE book ch. 12 |
| Prefer realism over isolation | SWE book ch. 13; Fowler, mocks aren't stubs: https://martinfowler.com/articles/mocksArentStubs.html |
| A thin facade over a third-party API | Hynek, what to mock: https://hynek.me/articles/what-to-mock-in-5-mins/ |
| `atol` near zero | `pytest.approx`: https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx |
| Seeded streams may change | NumPy compatibility policy: https://numpy.org/doc/stable/reference/random/compatibility.html |
| `--durations` | https://docs.pytest.org/en/stable/how-to/usage.html |

## Repo decisions

- The worktree, `uv sync` and `-m "not render"` mechanics; render tests are asked for, not run.
- Justify every tolerance: `AGENTS.md`'s rule that every physical number cites a source.
- Test Blender cheaply: `AGENTS.md`'s "Where code goes".
- A test never restates config.
- The findings table, shared by every reviewer.

## Judgement

- A test pays only for what it checks.
- Every behaviour the change adds or fixes has a test.
- Test a formula by limits, invariants and an independent reference.
- Golden images only on tiny fixtures.

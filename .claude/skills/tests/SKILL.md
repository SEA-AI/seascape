---
name: tests
description: >-
  Decides what a pytest test in `tests/` checks and what it costs: dead tests, `parametrize` over loops, fixtures and real collaborators, tolerances, render tests. Use before writing or changing a test, even mid-task, and whenever a change adds or fixes behaviour a test should pin; `review-pr` runs it as one reviewer. The idiom of the code under test is `stack`'s.
---

# Tests

## Does it fail when the code is wrong?

- **Break it and run.** See each test the change adds or relies on fail with its behaviour broken. Undoing purely added code only proves an `ImportError`. For each behaviour the change adds or fixes:
  1. `git worktree add --detach "${TMPDIR:-/tmp}/<name>" HEAD`, then `uv sync` in it. Never edit the PR's own worktree: a break left there can be committed.
  2. Break the behaviour there: revert the fix, or change the condition or value it adds.
  3. `uv run pytest <test ids> -m "not render"` on the tests that claim it, without `-x`, so every one runs. A test that still passes does not pin the behaviour it claims. A render test takes minutes, so it is reported not run, with its command; when the behaviour is the sea shader's, run its render check as AGENTS.md asks.
  4. `git worktree remove --force "${TMPDIR:-/tmp}/<name>"`.

  On a re-review, run only the tests the fix touched.
- **Every behaviour the change adds or fixes has a test**, or the finding names the smallest one that would pin it.
- **Keep only the tests a break needs.** Run every break against every test the change adds, by test id: `-k` can skip one by its name. A test or parametrized case is needed when some break fails it alone; before calling one dead, look for a break only it would fail. A fixture that raises runs no assertion, so its error shows the test was reached, not that it catches the break. A test not run, such as a render test, is unresolved, not dead.
- **A test never restates config.** Asserting that a default, a preset or a TOML field equals the literal written there checks nothing and changes with it. Test what the value does: not `assert Sea().wind_speed_mps == 7.0`, but the sea that wind builds.

## What it checks

- **Test what the code promises:** outputs, visible state, an invariant or a published relation with its source. Do not assert private names, call order, mock arguments, log wording or tuned numbers: they change when the behaviour does not, so the test fails for the wrong reason.
- **One reason to fail per test**, a name that states the behaviour. A table of cases is `parametrize`, never a `for` or `if` around an assert. Prefer DAMP to DRY: a test reads top to bottom.
- **Prefer real collaborators.** Double only what is slow, nondeterministic or foreign, and wrap a third-party API in a thin facade you own rather than mocking its objects.
- **Test a physical formula** by analytic limits, invariants (symmetry, monotonicity, scaling), and an independent reference. Justify every tolerance in a comment and set `atol` explicitly for values near zero. Golden images only on tiny fixtures with a tolerance.
- **Test statistics and properties of random output, not exact draws:** a seeded stream may change between NumPy releases.
- **Test Blender cheaply:** the pure layer needs no Blender. Assert graph structure on a fresh scene. Evaluate a graph with one small probe render per property and compare with the pure function's value. Share one built scene across tests.

## What it costs

- **A test pays only for what it checks.** Run `pytest --durations=10` on the tests the change touches, then cut what a slow one doesn't need: a smaller unit, a smaller input (lower resolution, fewer samples, fewer frames), a setup shared with a test that already pays for it.

## Reviewing

More tests are not more safety. Two that fail on the same breaks cost twice to read and keep for one behaviour pinned, so every test the change adds earns its place with a break only it fails.

## What a finding contains

Report the break matrix first, one row per break and the tests it fails, then a table, dead tests first:

| # | Where | Verdict | Finding | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | `tests/test_x.py:42` | dead | passes with the behaviour broken | ran it with the fix reverted | assert the keyed yaw steps |

- **Verdict:** **dead**, **missing**, **fix** or **keep**. A keep gives its reason in Fix.
- **Evidence:** what you ran and saw, or "not run".

End with what you did not run.

Sources are in [references/sources.md](references/sources.md).

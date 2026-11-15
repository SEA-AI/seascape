---
name: tests
description: What a seascape test checks and what it costs. Use when writing or changing a test, or when a change adds behaviour a test should pin; `review-pr` runs it as one reviewer. The idiom of the code under test is `stack`'s.
---

# Tests

## Does it fail when the code is wrong?

- **Revert and run.** For each behaviour the change adds or fixes, undo its hunk in a scratch worktree (`git worktree add --detach "$TMPDIR/<name>" HEAD`, then `uv sync` in it), run the test that claims it with `uv run pytest <test ids> -x -m "not render"`, and see it fail. A test that still passes is dead. A render test is reported not run, with its command. Remove the worktree after with `git worktree remove --force`. On a re-review, run only the tests the fix touched. Never edit the PR's own worktree.
- **Every new branch, error path and state has a test**, or the finding names the smallest one that would pin it.
- **A test never restates config.** Asserting that a default, a preset or a TOML field equals the literal written there checks nothing and changes with it. Test what the value does.

## What it checks

- **Test what the code promises:** outputs, visible state, an invariant or a published relation with its source. Do not assert private names, call order, mock arguments, log wording or tuned numbers.
- **One reason to fail per test**, a name that states the behaviour. A table of cases is `parametrize`, never a `for` or `if` around an assert. DAMP beats DRY: a test reads top to bottom.
- **Prefer real collaborators.** Double only what is slow, nondeterministic or foreign, and wrap a third-party API in a thin facade you own rather than mocking its objects.
- **Test a physical formula** by analytic limits, invariants (symmetry, monotonicity, scaling), and an independent reference. Justify every tolerance in a comment and set `atol` explicitly for values near zero. Golden images only on tiny fixtures with a tolerance.
- **Test statistics and properties of random output, not exact draws:** a seeded stream is not stable across NumPy releases.
- **A vectorised function keeps the loop it replaced as its test oracle.**
- **Test Blender cheaply:** the pure layer needs no Blender. Assert graph structure on a fresh scene. Evaluate a graph with one small probe render per property and compare with the pure function's value. Share one built scene across tests.

## What it costs

- **A test pays only for what it checks.** Run `pytest --durations=10` on the tests the change touches, then cut what a slow one doesn't need: a smaller unit, a smaller input (lower resolution, fewer samples, fewer frames), a setup shared with a test that already pays for it. A test that probes only the sea or sky builds `scenarios/open-sea.toml`, which imports no hull.
- **A helper, fixture or `conftest` layer used by fewer than three tests** is not yet worth its indirection.

## What a finding contains

Report a table, dead tests first:

| # | Where | Check | Evidence | Fix |
|---|---|---|---|---|
| 1 | `tests/test_x.py:42` | fails when the code is wrong | passes with the fix reverted | assert the keyed yaw steps |

- **Evidence:** what you ran and saw, or "not run".

End with what you did not run.

## Sources

- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- Hynek, what to mock: https://hynek.me/articles/what-to-mock-in-5-mins/
- Fowler, mocks aren't stubs: https://martinfowler.com/articles/mocksArentStubs.html

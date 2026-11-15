---
name: tests
description: What a test checks and what it costs. Use when writing or changing a test, or when a change adds behaviour a test should pin; `review-pr` runs it as one reviewer. The idiom of the code under test is `stack`'s.
---

# Tests

## Does it fail when the code is wrong?

- **Break it and run.** See each test the change adds or relies on fail with its behaviour broken. Undoing purely added code only proves an `ImportError`. For each behaviour the change adds or fixes:
  1. `git worktree add --detach "$TMPDIR/<name>" HEAD`, then `uv sync` in it. Never edit the PR's own worktree.
  2. Break the behaviour there: revert the fix, or change the condition or value it adds.
  3. `uv run pytest <test ids> -x -m "not render"` on the tests that claim it. A test that still passes is dead. A render test is reported not run, with its command.
  4. `git worktree remove --force "$TMPDIR/<name>"`.

  On a re-review, run only the tests the fix touched.
- **Every behaviour the change adds or fixes has a test**, or the finding names the smallest one that would pin it.
- **A test never restates config.** Asserting that a default, a preset or a TOML field equals the literal written there checks nothing and changes with it. Test what the value does.

## What it checks

- **Test what the code promises:** outputs, visible state, an invariant or a published relation with its source. Do not assert private names, call order, mock arguments, log wording or tuned numbers.
- **One reason to fail per test**, a name that states the behaviour. A table of cases is `parametrize`, never a `for` or `if` around an assert. Prefer DAMP to DRY: a test reads top to bottom.
- **Prefer real collaborators.** Double only what is slow, nondeterministic or foreign, and wrap a third-party API in a thin facade you own rather than mocking its objects.
- **Test a physical formula** by analytic limits, invariants (symmetry, monotonicity, scaling), and an independent reference. Justify every tolerance in a comment and set `atol` explicitly for values near zero. Golden images only on tiny fixtures with a tolerance.
- **Test statistics and properties of random output, not exact draws:** a seeded stream may change between NumPy releases.
- **Test Blender cheaply:** the pure layer needs no Blender. Assert graph structure on a fresh scene. Evaluate a graph with one small probe render per property and compare with the pure function's value. Share one built scene across tests.

## What it costs

- **A test pays only for what it checks.** Run `pytest --durations=10` on the tests the change touches, then cut what a slow one doesn't need: a smaller unit, a smaller input (lower resolution, fewer samples, fewer frames), a setup shared with a test that already pays for it.

## What a finding contains

Report a table, dead tests first:

| # | Where | Verdict | Finding | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | `tests/test_x.py:42` | dead | passes with the behaviour broken | ran it with the fix reverted | assert the keyed yaw steps |

- **Verdict:** **dead**, **missing**, **fix** or **keep**. A keep gives its reason in Fix.
- **Evidence:** what you ran and saw, or "not run".

End with what you did not run.

Sources are in [references/sources.md](references/sources.md).

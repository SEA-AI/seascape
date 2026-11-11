---
name: prose
description: Write or review seascape prose: comments, docstrings, docs, `assets.toml` notes, commit messages and PR bodies. Use whenever a change adds or edits any of them, and to review a PR's text. Code logic is out of scope.
---

# Prose

Every line of prose must lose something when deleted.

## Rules

1. Try deleting it first. For every line, ask whether anything is lost without it. If not, delete it. Rephrase only lines that must stay.
2. A rephrase is never longer than the line it replaces, unless it corrects a stale fact.
3. Comments only say what the code can't: a non-obvious constraint, a unit, a workaround and what removes it, a source. Never the line restated. A better name beats an explanatory comment.
4. Nothing that can go stale. No numbers derived from values elsewhere, lists copied from another file, test names, or mentions of PRs, reviews, conversations or history. No hardware, specs or consumers the repo doesn't model. If a claim matters, a test asserts it.
5. A number's source sits beside it, in a comment: a published relation, a derivation from one, or a pinned measurement of Blender. A judgement says "Judgement:".
6. No machine cadence: no "X, not just Y" emphasis, lists of exactly three, uniformly long sentences, hedges like "it's worth noting", em-dash habits or tacked-on "-ing" clauses. Prefer the specific: a number, a file name or a flag.
7. State the fact, not the journey. No debugging narration, no restating the diff, no closing recap.
8. A docstring matches its function's current parameters: a backticked name of a removed or folded-in parameter is stale. Each `Args:` entry teaches something the name does not: what the quantity is, its unit, a typical range, its source. Single backticks for names; no Sphinx roles or directives.

## Commits and pull requests

- A commit is a Conventional Commit (`feat:`, `fix:`, `chore:`, `ci:`) with an imperative subject; the body says why. Tidying and behaviour change are separate commits.
- A PR body is What, Why, a short How and Testing. It describes only its final state against its base, never how it evolved inside the PR: no "now" or "no longer" about an intermediate commit.
- Testing holds evidence CI and the diff do not show: a measurement, a cost, a render. An image wherever the change is visible; any before/after compares the base with the head.
- Prefer a table, a before/after or a frame over a paragraph. An outside source behind a number or choice gets one line.

## Reviewing

Review only the prose in the final diff, commit messages and PR body. Output a list of proposed edits, each as `file:line` (a commit by its hash), then old -> new (or "delete"). End with the total net prose lines: added minus removed.

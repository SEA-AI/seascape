---
name: prose
description: Write or review seascape prose: comments, docstrings, docs, `assets.toml` notes, commit messages and PR bodies. Use whenever a change adds or edits any of them; `review-pr` runs it as one reviewer. Code logic is out of scope.
---

# Prose

Delete first. A line stays only if the reader loses something without it.

## Delete

- **Restating the code.** A comment or docstring that says what the next line does, or what its name already says. A better name beats the comment.
- **Restating the diff.** A PR body or commit that lists the files touched or walks through the change line by line.
- **Context spill.** The journey: debugging narration, what was tried, reviews, conversations, other PRs, "now" or "no longer" about an earlier state.
- **Redundancy.** The same fact in two places, a summary of what was just said, a closing recap, an intro that announces what follows.
- **Verbosity.** Hedges ("it's worth noting"), filler, "X, not just Y" emphasis, lists padded to three, tacked-on "-ing" clauses, em-dash habits.
- **What goes stale.** Numbers derived from values elsewhere, lists copied from another file, test names, hardware, specs or consumers the repo doesn't model. If a claim matters, a test asserts it.

## Keep

- What the code can't say: a non-obvious constraint, a unit, a workaround and what removes it.
- A number's source beside it: a published relation, a derivation from one, or a pinned measurement of Blender. A judgement says "Judgement:".
- The specific over the general: a number, a file name, a flag.
- A table, a before/after or a frame over a paragraph.

A rephrase is never longer than what it replaces, unless it corrects a stale fact.

## Where it applies

- **Comments:** the why, never the what.
- **Docstrings:** match the current signature; a backticked name that no longer exists is stale. Each `Args:` entry says what the name does not: the quantity, its unit, a typical range, its source. Single backticks; no Sphinx roles.
- **Docs:** how to use the thing as it is, not how it came to be.
- **Commits:** a Conventional Commit (`feat:`, `fix:`, `chore:`, `ci:`), imperative subject, a body that says why. Tidying and behaviour change are separate commits.
- **PR bodies:** the final state against the base. Their shape is the PR template's.

## Reviewing

Review the prose in the final diff, the commit messages and the PR body. Be picky: every line added or changed gets a verdict, and when in doubt the verdict is delete.

Report a table, one row per line or passage, deletes first:

| # | Where | Verdict | Old -> new | Why |
|---|---|---|---|---|
| 1 | `file:line` | delete | `old` | restates the code |
| 2 | `body:3` | rephrase | `old` -> `new` | verbose |
| 3 | `abc1234` | keep | | the unit's only source |

- **Where:** `file:line`, a commit's hash, or `body:line`.
- **Verdict:** delete, rephrase or keep. A keep names what the reader would lose.
- **Why:** the category from Delete or Keep above, in a few words.

End with the net prose lines: added minus removed, after the edits.

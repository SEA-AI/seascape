---
name: review-pr
description: Review a pull request or branch diff of this repo with fresh reviewers (correctness, architecture, stack, tests, prose, and frames when it applies). Use when asked to review a PR or branch, before handing a PR over, and again after a push that adds prose.
---

# Review a pull request

Every PR gets these reviews before it is handed over. Each reviewer is a fresh subagent that did not write the change. Give each one the PR's final diff against its base, its commit messages and its body, and say that it edits nothing. Run them in parallel, one PR at a time.

## The reviewers

1. **Correctness.** Run `/code-review` on the PR. Point it at `AGENTS.md`'s traps that fail silently first.
2. **Architecture.** Follow the `architecture` skill on the diff.
3. **Stack.** Follow the `stack` skill on the diff.
4. **Tests.** Follow the `tests` skill on the diff.
5. **Prose.** Follow the `prose` skill.
6. **Frames,** when the `frames` skill's description applies to the diff. Follow the `frames` skill on the diff.

## After the reviews

A review request ends with the findings, merged into one table:

| # | Reviewer | Where | Verdict | Finding | Fix | Outcome |
|---|---|---|---|---|---|---|
| 1 | correctness | `file:line` | bug | a stale `matrix_world` | `view_layer.update()` first | applied |
| 2 | architecture | `file:line` | replace | a hand-rolled clamp | `np.clip` | applied |
| 3 | prose | `body:4` | delete | restates the diff | | rejected: the reason |

- **Order:** bugs and dead tests, then deletes and replaces, then the rest; keeps last.
- **Finding:** what is wrong, with the cost or evidence where the reviewer gives them. A keep's reason goes in Fix.

Edit and push only when handing over a PR you wrote, or when asked to fix; until then Outcome is blank, and the steps below do not apply.

- For each finding but a keep, apply its Fix or give a one-line reason a reader can check for rejecting it.
- Run the tests the fixes touch, then push with the `gh-stack` skill.
- After a fix that changes code, run every review but prose again on the final diff, and prose too when the fix adds a comment or docstring.

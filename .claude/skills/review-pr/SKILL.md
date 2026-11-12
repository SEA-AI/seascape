---
name: review-pr
description: Review a seascape pull request or diff with five fresh reviewers (correctness, architecture, anti-hack, stack, prose). Use for any review request, before handing a PR over, and again after a push that adds prose.
---

# Review a pull request

Every PR gets five reviews before it is handed over. Each reviewer is a fresh subagent that did not write the change. Give each one the PR's final diff against its base, its commit messages and its body, and say that it edits nothing. Run them in parallel, one PR at a time.

## The reviewers

1. **Correctness.** Run `/code-review` on the PR. Point it at `AGENTS.md`'s traps that fail silently first.
2. **Architecture.** Follow the `architecture` skill on the diff.
3. **Anti-hack.** Follow the `anti-hack` skill on the diff.
4. **Stack.** Follow the `stack` skill on the diff.
5. **Prose.** Follow the `prose` skill.

## After the reviews

A review request ends with the findings, merged into one table:

| # | Reviewer | Where | Verdict | Finding | Fix | Outcome |
|---|---|---|---|---|---|---|
| 1 | correctness | `file:line` | bug | a stale `matrix_world` | `view_layer.update()` first | applied |
| 2 | anti-hack | `file:line` | replace | a hand-rolled clamp | `np.clip` | applied |
| 3 | prose | `body:4` | delete | restates the diff | | rejected: the reason |

- **Order:** bugs, then deletes and replaces, then the rest; keeps last.
- **Verdict:** the reviewer's own; `—` where its table has none, as stack's.
- **Finding:** what is wrong, and its cost where the reviewer gives one. A keep's reason goes here.
- **Fix:** the replacement, root fix or new text the reviewer names.

Edit and push only when handing over a PR you wrote, or when asked to fix; until then Outcome is blank, and the steps below do not apply.

- Apply the fixes each skill's verdicts name. A finding you reject gets a one-line reason a reader can check.
- Run the tests the fixes touch, then push through `gh stack`.
- After a fix that changes code, run the correctness, architecture, anti-hack and stack reviews again on the final diff.

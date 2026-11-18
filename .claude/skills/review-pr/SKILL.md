---
name: review-pr
description: Reviews a pull request or branch diff of this repo with fresh subagent reviewers (correctness, architecture, stack, tests, prose, and frames when it applies). Use when asked to review a PR or branch, before handing a PR over, and after a push that adds code or prose.
---

# Review a pull request

Every PR gets these reviews before it is handed over. Each reviewer is a fresh subagent that did not write the change. Give each one the PR's final diff against its base, its commit messages and its body. Run them in parallel, one PR at a time.

A brief for reviewers 2 to 6: You are the `<skill>` reviewer. Read `.claude/skills/<skill>/SKILL.md` and follow it on the diff against the base (`gh pr diff <N>` or `git diff <base>...HEAD`), `git log <base>..HEAD` and the PR body. Edit nothing. Return the findings table and what you did not read.

## The reviewers

1. **Correctness.** Run `/code-review medium` on the PR, or at the level the user names: without one, it reuses the last level typed. Point it at `AGENTS.md`'s traps that fail silently first.
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
- Run the tests the fixes touch, then push with the `gh-stack` skill (`gh skill install github/gh-stack`).
- After a fix that changes code, run every review but prose again on the final diff, and prose too when the fix adds a comment or docstring.

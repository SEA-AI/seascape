---
name: review-pr
description: Review a seascape pull request or diff with five fresh reviewers (correctness, design, anti-hack, stack, prose), then fix what they find. Use for any review request, before handing a PR over, and again after a push that adds prose.
---

# Review a pull request

Every PR gets five reviews before it is handed over. Each reviewer is a fresh subagent that did not write the change. Give each one the PR's final diff against its base, its commit messages and its body, and say that it edits nothing. Run them in parallel, one PR at a time.

## The reviewers

1. **Correctness.** Run `/code-review` on the PR. Point it at `AGENTS.md`'s traps that fail silently first.
2. **Design.** Follow the `design` skill on the diff.
3. **Anti-hack.** Follow the `anti-hack` skill on the diff.
4. **Stack.** Follow the `stack` skill on the diff.
5. **Prose.** Follow the `prose` skill.

## After the reviews

- Apply the fixes each skill's verdicts name. A finding you reject gets a one-line reason a reader can check.
- Run the tests the fixes touch, then push through `gh stack`.

---
name: review-pr
description: Review a seascape pull request with five fresh reviewers (correctness, design, anti-hack, Python practice, prose), then fix what they find. Use before handing any PR over, and again after a push that adds prose.
---

# Review a pull request

Every PR gets five reviews before it is handed over. Each reviewer is a fresh subagent that did not write the change. Give each one the PR's final diff against its base, its commit messages and its body, and say that it edits nothing. Run them in parallel, one PR at a time.

## The reviewers

1. **Correctness.** Run `/code-review` on the PR. Point it at `AGENTS.md`'s traps that fail silently first.
2. **Design.** Follow the `design` skill on the diff.
3. **Anti-hack.** Follow the `anti-hack` skill on the diff.
4. **Python practice.** Follow the `clean-code` skill and its `references/stack.md` on the diff. Any test it proposes is the smallest one that pins the behaviour.
5. **Prose.** Follow the `prose` skill on the diff, the commit messages and the PR body.

## After the reviews

- Fix every finding at its root, in the layer that owns it. A finding you reject gets a one-line reason a reader can check.
- A real problem outside the PR's scope becomes an issue, not a silent skip.
- Run the tests the fixes touch, then push through `gh stack`.
- Run the prose review again after any push that adds text.

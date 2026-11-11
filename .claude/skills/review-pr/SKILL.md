---
name: review-pr
description: Review a seascape pull request with four fresh reviewers (correctness, architecture, Python practice, prose), then fix what they find. Use before handing any PR over, and again after a push that adds prose.
---

# Review a pull request

Every PR gets four reviews before it is handed over. Each reviewer is a fresh subagent that did not write the change. Give each one the PR's final diff against its base, its commit messages and its body, and say that it edits nothing. Run the four in parallel, one PR at a time.

## The reviewers

1. **Correctness.** Run `/code-review` on the PR. Point it at `AGENTS.md`'s traps that fail silently first.
2. **Architecture.** Follow the `review-architecture` skill on the diff. For every addition, ask whether it should have been a refactor, whether a tool or existing code already does it, and whether it is a workaround with no documented constraint.
3. **Python practice.** Follow the `clean-code` skill and its `references/stack.md` on the diff. Any test it proposes is the smallest one that pins the behaviour.
4. **Prose.** Give the brief below, word for word.

## The prose brief

> Review only the prose in the final diff: comments, docstrings, docs, `assets.toml` notes, commit messages and the PR body. Leave code logic alone.
> 1. Try deleting it first. For every added line of prose, ask whether anything is lost without it. If not, delete it. Rephrase only lines that must stay.
> 2. A rephrase is never longer than the line it replaces.
> 3. Report net lines for the prose: added minus removed.
> 4. Comments only say what the code can't. A comment should give a non-obvious constraint, a unit, a workaround or a source. It should never restate the line. A better name beats an explanatory comment.
> 5. Nothing that can go stale. Cut numbers derived from values elsewhere, lists copied from another file, test names, and mentions of PRs, reviews, conversations or history. Also cut hardware, specs or consumers the repo doesn't model.
> 6. Every physical number cites its source beside it: a published relation, a derivation from one, or a pinned measurement of Blender. Never a constant fitted to a render.
> 7. No machine cadence. Remove "X, not just Y" emphasis, lists of exactly three, uniformly long sentences, hedges like "it's worth noting", em-dash habits and tacked-on "-ing" clauses. Prefer the specific: a number, a file name or a flag.
> 8. State the fact, not the journey. No debugging narration, no restating the diff, no closing recap paragraph.
> 9. The PR describes only its final state against its base. Never how it evolved inside the PR: no "now", "no longer" or "fixed the step" about an intermediate commit, and every before/after image or plot compares the base branch with the head, never an earlier commit of the same PR.
> 10. Check every docstring the diff touches against its function's current parameters: a backticked name of a removed or folded-in parameter is stale.
> Output: a list of proposed edits, each as `file:line`, then old -> new (or "delete"), then the net line count.

## After the reviews

- Fix every finding at its root, in the layer that owns it. A finding you reject gets a one-line reason a reader can check.
- A real problem outside the PR's scope becomes an issue, not a silent skip.
- Run the tests the fixes touch, then push through `gh stack`.
- Run the prose review again after any push that adds text.

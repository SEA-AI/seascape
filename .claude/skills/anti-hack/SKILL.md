---
name: anti-hack
description: Whether a piece of seascape code should exist at all, and whether it is the proper mechanism or a workaround. Use before writing a helper, flag, special case, constant or retry, or to audit a package for reinvented helpers, uncited constants and dead code. Where code lives is `design`'s; bugs are `/code-review`'s; `review-pr` runs it as one reviewer.
---

# Anti-hack

Question every new line: is it justified, or an odd workaround?

## Questions for every addition

- **Does it need to exist?** Name who asked for it. An option, hook or generality nobody uses yet is inventory.
- **Does something already do it?** Blender (a node, a render setting, an operator), the standard library, numpy, cv2, Pillow, pydantic or this repo. Name the call that does the same job.
- **Is it a patch over the wrong mechanism?** A special case, flag, tuning knob, global, monkeypatch, lint suppression, sleep or retry, or a `try` around a symptom. Find the simpler formulation, usually an established pattern with a precedent. Judge by the simplest resulting model, not the smallest diff.
- **Is every number sourced?** A constant with no source beside it, one fitted to a render, or one that repeats a value a preset, Blender or another module already holds.
- **Is it dead?** Unused code, commented-out code, a function only tests call, a flag or shim without a removal condition.

Only a documented constraint justifies an odd shape: an `AGENTS.md` trap, a Blender limit or a measured cost. That the codebase does it elsewhere is never a reason; it is a second finding.

## A limitation found

Spend a real attempt on the fix first; the repo often holds the data that removes the need for a workaround. If it does not fit the change, fix it in its own change or open an issue. Never leave it to memory. Mark a kept limitation in code with `Limitation:` and the condition that removes it.

## Auditing a package

1. Read `AGENTS.md` and write the scope and skip list down first. Audit the tip of the branch the fixes land on.
2. One reader per package, with the same brief and compact output. Read whole files, not excerpts.
3. Before calling code unused, search `seascape/`, `tests/`, `docs/`, the TOML files and string references.
4. Before accepting a replacement, run old and new on random inputs and compare, or render both and compare pixels against the render's own noise floor. Say "unverified" where you did not.
5. Count the helper a fix adds: a dedupe that adds as much as it removes is not a finding.
6. Regenerate committed outputs, such as doc images, and diff them; skip a fix that alters them.
7. Record each keep with its reason in the report, so nobody audits it twice.

## What a finding contains

- `file:line`, and what it re-does or works around.
- The replacement: the existing call, the proper mechanism, or the source for the number.
- Net lines the fix removes.
- A verdict: **replace**, **delete**, **keep (reason)** or **open an issue**.

Number the findings and rank them by net lines removed. End with what you did not read.

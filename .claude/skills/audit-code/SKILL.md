---
name: audit-code
description: Audit seascape for code that reinvents what Blender, the standard library, numpy, cv2 or pydantic already does, for uncited or duplicated constants, and for dead or test-only code. Use when asked "how much of this is reinvented?", before a "less is more" cleanup, or when a review finds one hand-rolled helper and you want to know how many more exist.
---

# Audit code

An audit answers one question with evidence: what here should not exist, and what replaces it. It edits nothing. Fixes come after, as separate, small changes.

## Set up

1. Read `AGENTS.md`. Its rule ("Blender does anything it can do. Code is a liability.") and its documented designs decide what is a finding. A design it documents is not.
2. Write the scope and the skip list down before reading: which packages, which files are already being reworked, which constants already cite a source.
3. Audit the tip of the branch the fixes will land on. Line numbers and files drift between `main` and an in-flight stack.

## What to look for

- **Reinvention.** Ask of every hand-written helper: does Blender (node, render setting, operator), the standard library, numpy, cv2, Pillow or pydantic do this? Typical hits are a manual loop for one numpy call, an f-string where a dict is accepted, a hand-written validator where a `Field` constraint exists, the same few lines pasted in three files.
- **Constants.** A number with no source beside it, or one that repeats a value a preset, Blender or another module already holds.
- **Dead and test-only code.** Grep for callers in `seascape/`, `tests/` and `docs/`. A function only tests call belongs in `tests/`.

## Method

- One reader per package, all with the same brief and the same compact output: `file:line`, what it re-does, the exact replacement, size, verdict. Read the whole file, not excerpts.
- Verdicts are **replace**, **keep (reason)** or **open an issue**. Record the keeps, so nobody audits them twice.
- Rank by net lines deleted. Count the helper you would add: a dedupe that adds as much as it removes is not a finding.
- Be sceptical of a hit. Look for why the code is the way it is: a hot loop, a dependency the project does not have (scipy), a behaviour the replacement lacks (cv2 and a BGRA frame), a trap `AGENTS.md` documents.

## Check before you trust

- Dead-code claims: grep them yourself.
- A replacement: run old and new on random inputs and compare, or render both and compare pixels. Say "unverified" where you did not.
- A constant: find its source or label it a judgement; never fit it to a render.
- Outputs that are committed, such as doc images: regenerate and diff them. Skip a change that would alter them.

## Report

Lead with the count and the biggest items. Then a ranked table (where, what, replacement, lines, verdict), a constants table and the keep list. Put long tables in the shared notes folder, not in a message.

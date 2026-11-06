---
name: audit-code
description: Audit seascape for code that reinvents what Blender, the standard library, numpy, cv2 or pydantic already does, for uncited or duplicated constants, and for dead or test-only code. Use before a "less is more" cleanup, or when a review finds one hand-rolled helper.
---

# Audit code

An audit edits nothing: it lists what should not exist and what replaces it. Fixes follow as separate changes.

## Set up

1. Read `AGENTS.md`: its rule defines a finding, a design it documents is not one.
2. Write the scope and the skip list down before reading: which packages, which files are already being reworked, which constants already cite a source.
3. Audit the tip of the branch the fixes land on; `main` drifts from an in-flight stack.

## What to look for

- **Reinvention.** Ask of every hand-written helper: does Blender (node, render setting, operator), the standard library, numpy, cv2, Pillow or pydantic do this?
- **Constants.** A number with no source beside it, or one that repeats a value a preset, Blender or another module already holds.
- **Dead and test-only code.** Grep for callers in `seascape/`, `tests/` and `docs/`. A function only tests call belongs in `tests/`.
- **Double guards.** A check Python, pydantic or the library already makes with a clear error.
- **Single-caller wrappers.** A helper with one caller is another place to look; inline it.
- **Said twice.** A comment that restates its line, or a fact written in two files.
- **Tests of wiring.** A test that fails when something moves but not when the number is wrong, or that compares one test helper with another.

## Method

- One reader per package, all with the same brief and the same compact output: `file:line`, what it re-does, the exact replacement, size, verdict. Read the whole file, not excerpts.
- Verdicts are **replace**, **keep (reason)** or **open an issue**. Record the keeps, so nobody audits them twice.
- Rank by net lines deleted. Count the helper you would add: a dedupe that adds as much as it removes is not a finding.
- Before accepting a hit, find why the code is as it is: a hot loop, a missing dependency, a behaviour the replacement lacks, a trap `AGENTS.md` documents.

## Check before you trust

- Dead-code claims: grep them yourself.
- A replacement: run old and new on random inputs and compare, or render both and compare pixels. Say "unverified" where you did not.
- A constant: find its source or label it a judgement; never fit it to a render.
- Outputs that are committed, such as doc images: regenerate and diff them. Skip a change that would alter them.

## Report

Lead with the count and the biggest items, then a ranked table (where, what, replacement, lines, verdict), a constants table and the keep list.

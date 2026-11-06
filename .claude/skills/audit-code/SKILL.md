---
name: audit-code
description: Audit seascape for code that reinvents what Blender, the standard library, numpy, cv2 or pydantic already does, for uncited or duplicated constants, and for dead or test-only code. Use before a "less is more" cleanup, or when a review finds one hand-rolled helper. Bugs, security and load are `ponytail-audit`'s.
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
- **Dead and test-only code.** A function only tests call belongs in `tests/`.
- **Smells.** Double guards, pass-through wrappers, comments that restate, tests of wiring: `clean-code`'s "Smells, as questions".

## Method

- One reader per package, all with the same brief and the same compact output: `file:line`, what it re-does, the exact replacement, size, verdict. Read the whole file, not excerpts.
- Verdicts are **replace**, **keep (reason)** or **open an issue**. Record the keeps, so nobody audits them twice.
- Rank by net lines deleted. Count the helper you would add: a dedupe that adds as much as it removes is not a finding.
- Before accepting a hit, find why the code is as it is: a hot loop, a missing dependency, a behaviour the replacement lacks, a trap `AGENTS.md` documents.

## Check before you trust

- No concrete case, no finding: name the call that does the same job.
- Before calling code unused, grep `seascape/`, `tests/`, `docs/`, the TOML files and string references. A static finder misses dynamic use.
- A replacement: run old and new on random inputs and compare, or render both and compare pixels. A render with a fixed seed is bit-reproducible in EO; IR is not, so compare a run against itself first. Say "unverified" where you did not.
- A constant: find its source or label it a judgement; never fit it to a render.
- Outputs that are committed, such as doc images: regenerate and diff them. Skip a change that would alter them.

## Report

Number the findings, so "fix 2 and 5" works. Lead with the count and the biggest, then a ranked table (where, what, replacement, lines, verdict), a constants table and the keep list. End with the net lines removable and what you did not read.

---
name: architecture
description: Decides whether code should exist and which layer owns it (SOLID, information hiding, the import-linter layers). Use before writing a helper, flag, special case, constant, retry, module, record field, callback, Blender custom property or a second home for a formula; when choosing where code goes, when one feature touches many layers, before refactoring across layers, or to audit a package for reinvented helpers, uncited constants and dead code. `review-pr` runs it as one reviewer. Language and library practice is `stack`'s, comments `prose`'s, tests `tests`', bugs `/code-review`'s.
---

# Architecture

Each piece earns its place, lands where its decision lives, and the whole reads as a reusable library.

`AGENTS.md` and the import-linter contract record project decisions. A review may find one that a principle rejects: report it as an issue for the owner, not as a reason.

## Before writing

1. Ask "Should it exist?" below.
2. Place it on the map.
3. Count the homes of the formula or rule it touches.
4. Ask the principle questions below of each new module, union member, interface or call across layers.

## Should it exist?

- **Does it need to exist?** Name who asked for it. An option, hook or generality nobody uses yet is inventory.
- **Does something already do it?** Blender (a node, a render setting, an operator), the standard library, an installed package or this repo. Name the call that does the same job.
- **Is it a patch over the wrong mechanism?** A special case, flag, tuning knob, global, monkeypatch, lint suppression, sleep or retry, or a `try` around a symptom. Find the simpler formulation, usually an established pattern with a precedent. Judge by the simplest resulting model, not the smallest diff.
- **Is every physical number sourced,** as `AGENTS.md` asks? And does a constant repeat a value a preset, Blender or another module already holds?
- **Is it dead?** Unused code, commented-out code, a function only tests call, a flag or shim without a removal condition.

Only a documented constraint justifies an odd shape: an `AGENTS.md` trap, a Blender limit or a measured cost. That the codebase does it elsewhere is never a reason; it is a second finding.

A limitation found gets a real attempt at the fix first; the repo often holds the data that removes the need for a workaround. If it does not fit the change, fix it in its own change or open an issue, so it survives the session. Mark a kept limitation in code with `Limitation:` and the condition that removes it.

## The map

`pyproject.toml`'s import-linter contract lists the layers; read the names there. Each layer owns one kind of decision:

- The lowest layers own relations with a source and pure helpers: functions of values, no `bpy`.
- Above them, the configuration layers parse and normalise what the user can say: units, defaults, cross-field checks. Anything that needs a sourced relation or evaluates time is computed by the layer that owns that relation, from the model's fields.
- Pure computation over validated input sits next: where a hull is at a time, what a frame becomes.
- The Blender layers are the imperative shell. They turn values into datablocks, nodes and keyframes. A branch there may choose how Blender stores a value (keyed or constant, which channel); a branch that chooses which value, kind or object belongs below it.
- Entry points parse arguments, call and print. They hold no decision; code that drives `bpy` belongs in the shell.

Dependencies point inward, from the shell to the core, never the reverse (Martin's dependency rule). A violation means the code is in the wrong layer, or the contract is wrong; it never earns an exception. An exception in the contract (`ignore_imports`) is a finding.

## Questions per principle

- **Single responsibility (Martin: one reason to change; one actor).** Name the change that forces an edit here. Two unrelated ones is divergent change: split. One change that forces edits in many modules is shotgun surgery: a decision scattered from its home.
- **Information hiding (Parnas).** Which design decision does this module hide? A module cut by processing step (Ousterhout's temporal decomposition) leaks the same knowledge into every step. Cut by the decision likely to change.
- **Deep modules (Ousterhout).** Is the interface much smaller than the work behind it? A function whose callers carry an intermediate value from one call to the next, or recover a result it hid, has exposed its internals. Return the finished thing.
- **Open/closed.** Adding a kind should touch the union and that kind's own code. Behaviour that varies by kind is a method on the member or one dispatch beside the union. Count the places that branch on the kind: more than one is the smell.
- **Liskov.** Ask it of union members and of any callable type that several producers return: can every caller use every member without checking which one it got?
- **Interface segregation.** Does a caller depend on a type wider than the slice it uses, so unrelated changes reach it? Pass the narrowest existing type that carries what it reads: a sub-model, or a field when it reads one. Never break a cohesive record into loose primitives.
- **Dependency inversion (Martin).** When the core must call out, to Blender, a file or a clock, the core owns the interface it needs: a `Callable` type in the core's layer, which the shell implements. Better still, pass values so no port is needed.
- **Functional core, imperative shell (Bernhardt).** Could this branch run on plain values and be tested without Blender? Then it belongs below the shell.
- **Connascence (Page-Jones, Weirich).** Name the form and rate it: strength (name is the weakest, and dynamic is worse than static), degree (how many places), locality (how far apart). A string key shared across layers is connascence of name that no tool checks: one constant or one typed field. A key order is connascence of position: named fields. A formula in two places is connascence of algorithm: one home. Strong forms belong inside one function.
- **Package interface.** Does a module import another module's `_private` name? The name is either public, or the caller is in the wrong place. import-linter does not see this.
- **Error boundaries.** The shell turns an engine failure into a domain error; an entry point decides the exit code. No core function catches a shell's error.

## Smells

Each names its usual root fix.

- **Blender as a message bus.** A value written to a custom property, or computed by Blender, only so another layer reads it back. The reader receives the value from the computation that made it. Custom properties are for Blender itself or a user in Blender.
- **A closure or call that rebuilds.** A callback, lambda or second call whose job is to recompute what another layer already computed. Pass the result.
- **One action, several homes.** Two call sites that must change together (one rule, one Blender trap, one conversion) get one owner now. Look-alikes that may diverge wait for a third case (Fowler's rule of three).
- **A field or default for one caller.** Compute it at that call site, or ask why the record does not carry the real quantity.
- **A tag with one legal value.** A `Literal` or enum of one member is speculative generality: delete it. A tag the user writes in a file is part of its format, and adding one later breaks every file; that one may come first.
- **A literal in the shell.** A physical or design number written into Blender-side code is a second source of truth. It lives in the pure layer and is imported by name. A number that only addresses Blender, such as a channel index, stays in the shell.
- **Feature envy.** A function that reads mostly another module's record belongs beside that record.
- **Data clumps.** The same few values passed together through several signatures become one record, defined at or below the lowest layer that uses it.

## When principles disagree

| Tension | Settle it by |
|---|---|
| SRP against deep modules | Split on a decision that changes independently. Never split one decision across two shallow modules. |
| Deduplication against the wrong abstraction (Metz) | Merge what must change together. Leave look-alikes. If a shared helper grew a flag per caller, inline it and look again. |
| A variation point against YAGNI | It arrives with the second kind, in the same change. A hook for an imagined case is inventory. |
| Layer purity against a one-line shortcut | The layer wins. |
| Moving code against a small diff | Report the misplacement; the move is tidying, a commit of its own as `prose` says. |

## Reviewing

1. Read `AGENTS.md`, then the import-linter contract, and run `uvx --from import-linter lint-imports`. A failure is a finding. So is a package that no layer names: a passing run says nothing about it.
2. List each public name the diff adds, each new field, and each parameter added to an existing signature. Ask each the questions under "Should it exist?"; each must answer "who needs this, and why here?" in one sentence. No answer is a finding.
3. Place each on the map. Ask whether a lower layer could own it.
4. Trace each value the diff introduces or reroutes: where it is computed, where it is stored, where it is read. Every hop through Blender, a closure or a recomputation needs a reason.
5. Count homes. For each action, conversion and formula the diff touches, count the places that perform it.
6. Ask the principle questions where steps 1 to 5 found friction.
7. Before reporting, find why the code is as it is. "Should it exist?" says which reasons justify an odd shape.

To audit a whole package, follow [references/audit.md](references/audit.md).

## What a finding contains

Report a table, deletes and replaces first, then the finding whose fix removes the most coupling:

| # | Where | Verdict | Finding | Cost | Fix |
|---|---|---|---|---|---|
| 1 | `file:line` | replace | a hand-rolled clamp | the next edge case is fixed in one clamp, not the other | `np.clip` |
| 2 | `a.py:12`, `b.py:40` | merge | one formula, two homes | the next unit change drifts | one function in `physics` |

- **Where:** the symptom, and the code it duplicates or depends on.
- **Verdict:** **replace**, **delete**, **move**, **merge**, **split**, **pass** (hand over the value instead of recomputing or reading it back), **keep** or **open an issue**. A keep gives its reason in Fix.
- **Finding:** the smell or principle, named in the words above.
- **Cost:** the next change that will go wrong, or the two places that will drift. "Unclean" is not a cost.
- **Fix:** the existing call, the proper mechanism, the number's source, or the root fix and the layer it belongs in. A fix that adds a layer, flag or wrapper is suspect; prefer deleting, moving or passing a value.

End with what you did not read.

Sources are in [references/sources.md](references/sources.md).

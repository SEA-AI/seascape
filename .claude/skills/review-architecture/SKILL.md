---
name: review-architecture
description: Review a seascape diff or package for where each piece lives and why it exists, through SOLID, information hiding and the layer order. Use when a change adds a module, a record field, a callback, a Blender custom property or a second home for a formula; when one feature touched many layers; or before a refactor that moves code between layers. Line-level style and the choice of container are `clean-code`'s; reinvented helpers and constants are `audit-code`'s.
---

# Review architecture

Question every new line: is it justified, or an odd workaround? Each piece lands where its decision lives, and the whole reads as a reusable library.

This skill reviews placement and responsibility, and edits nothing. It states principles; the code under review is evidence, never the norm. `AGENTS.md` and the import-linter contract record project decisions. A review may find one that a principle rejects: report it as an issue for the owner, not as a reason.

## The map

`pyproject.toml`'s import-linter contract lists the layers; read the names there. Each layer owns one kind of decision:

- The lowest layers own relations with a source and pure helpers: functions of values, no `bpy`.
- Above them, the configuration layers parse and normalise what the user can say: units, defaults, cross-field checks. Anything that needs a sourced relation or evaluates time is computed by the layer that owns that relation, from the model's fields.
- Pure computation over validated input sits next: where a hull is at a time, what a frame becomes.
- The Blender layers are the imperative shell. They turn values into datablocks, nodes and keyframes. A branch there may choose how Blender stores a value (keyed or constant, which channel); a branch that chooses which value, kind or object belongs below it.
- Entry points parse arguments, call and print. They hold no decision; code that drives `bpy` belongs in the shell.

Dependencies point from volatile code to stable code: the shell depends on the core, never the reverse (Martin's stable dependencies). A violation means the code is in the wrong layer, or the contract is wrong; it never earns an exception.

## Questions per principle

Ask each where the procedure below found friction.

- **Single responsibility (Martin: one reason to change; one actor).** Name the change that forces an edit here. Two unrelated ones is divergent change: split. One change that forces edits in many modules is shotgun surgery: a decision scattered from its home.
- **Information hiding (Parnas).** Which design decision does this module hide? A module cut by processing step (Ousterhout's temporal decomposition) leaks the same knowledge into every step. Cut by the decision likely to change.
- **Deep modules (Ousterhout).** Is the interface much smaller than the work behind it? A function whose callers carry an intermediate value from one call to the next, or recover a result it hid, has exposed its internals. Return the finished thing.
- **Open/closed.** Adding a kind should touch the union and that kind's own code. Behaviour that varies by kind is a method on the member or one dispatch beside the union. Count the places that branch on the kind: more than one is the smell.
- **Liskov.** Ask it of union members and of any callable type that several producers return: can every caller use every member without checking which one it got?
- **Interface segregation.** Does a caller depend on a type wider than the slice it uses, so unrelated changes reach it? Pass the narrowest existing type that carries what it reads: a sub-model, or a field when it reads one. Never break a cohesive record into loose primitives.
- **Dependency inversion (Martin).** When the core must call out, to Blender, a file or a clock, the core owns the interface it needs: a `Callable` type in the core's layer, which the shell implements. Better still, pass values so no port is needed.
- **Functional core, imperative shell (Bernhardt).** Could this branch run on plain values and be tested without Blender? Then it belongs below the shell.
- **Connascence (Page-Jones, Weirich).** Name the form and rate it: strength (name < type < meaning < position < algorithm, dynamic worse than static), degree (how many places), locality (how far apart). A string key shared across layers is connascence of name that no tool checks: one constant or one typed field. A key order is connascence of position: named fields. A formula in two places is connascence of algorithm: one home. Strong forms belong inside one function.
- **Package interface.** Does a module import another module's `_private` name? The name is either public, or the caller is in the wrong place. import-linter does not see this.
- **Error boundaries.** The shell turns an engine failure into a domain error; an entry point decides the exit code. No core function catches a shell's error.

## Smells

Each names its usual root fix.

- **Blender as a message bus.** A value written to a custom property, or computed by Blender, only so another layer reads it back. The reader receives the value from the computation that made it. Custom properties are for Blender itself or a user in Blender.
- **A closure or call that rebuilds.** A callback, lambda or second call whose job is to recompute what another layer already computed. Pass the result.
- **One action, several homes.** Two call sites that must change together (one rule, one Blender trap, one conversion) get one owner now. Look-alikes that may diverge wait for a third case (Fowler's rule of three).
- **A field or default for one caller.** Compute it at that call site, or ask why the record does not carry the real quantity.
- **A tag with one legal value.** A `Literal` or enum of one member is speculative generality: delete it. A tag the user writes in a file is part of its format, and adding one later breaks every file; that one may come first.
- **Feature envy.** A function that reads mostly another module's record belongs beside that record.
- **Data clumps.** The same few values passed together through several signatures become one record, defined at or below the lowest layer that uses it.

## When principles disagree

| Tension | Settle it by |
|---|---|
| SRP against deep modules | Split on a decision that changes independently. Never split one decision across two shallow modules. |
| Deduplication against the wrong abstraction (Metz) | Merge what must change together. Leave look-alikes. If a shared helper grew a flag per caller, inline it and look again. |
| A variation point against YAGNI | It arrives with the second kind, in the same change. A hook for an imagined case is inventory. |
| Layer purity against a one-line shortcut | The layer wins. |
| Moving code against a small diff | Report the misplacement. Move it in its own commit, never inside a behaviour change. |

## Review procedure

1. Read `AGENTS.md`, then the import-linter contract, and run `uvx --from import-linter lint-imports`. A failure is a finding. So is a package that no layer names: a passing run says nothing about it.
2. List each public name the diff adds, each new field, and each parameter added to an existing signature. For a package review, list its public names. Each must answer "who needs this, and why here?" in one sentence. No answer is a finding.
3. Place each on the map. Ask whether a lower layer could own it, and whether something already does: search for the quantity it computes, not its name.
4. Trace each value the diff introduces or reroutes: where it is computed, where it is stored, where it is read. Every hop through Blender, a closure or a recomputation needs a reason.
5. Count homes. For each action, conversion and formula the diff touches, count the places that perform it.
6. Ask the principle questions only where steps 1 to 5 found friction. Do not grade the diff against every principle.
7. Before reporting, find why the code is as it is. Only a documented constraint justifies an odd shape: an `AGENTS.md` trap, a Blender limit or a measured cost. Then it is a keep, with that reason. That the codebase does it elsewhere is never a reason; it is a second finding.

## What a finding contains

- `file:line` of the symptom, and of the code it duplicates or depends on.
- The smell or principle, named in the words above.
- Its cost: the next change that will go wrong, or the two places that will drift. "Unclean" is not a cost.
- The root fix and the layer it belongs in. A fix that adds a layer, flag or wrapper is suspect; prefer deleting, moving or passing a value.
- A verdict: **move**, **merge**, **split**, **pass** (hand over the value instead of recomputing or reading it back), **delete**, **keep (reason)** or **open an issue**.

Number the findings. Lead with the one whose fix removes the most coupling. End with what you did not read.

Sources are in [references/sources.md](references/sources.md).

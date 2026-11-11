---
name: review-architecture
description: Review a seascape diff or package for where each piece lives and why it exists, through SOLID, information hiding and the layer order. Use when a change adds a module, a record field, a callback, a Blender custom property or a second home for a formula; when one feature touched many layers; or before a refactor that moves code between layers. Line-level style is `clean-code`'s; reinvented helpers and constants are `audit-code`'s.
---

# Review architecture

Question every new line: is it justified, or an odd workaround? Every piece lands in the right place, and the whole reads as a cohesive, reusable library where each component and abstraction has its place.

This skill reviews placement and responsibility. It edits nothing. For naming, types, errors and tests use `clean-code`; for code that Blender, numpy or pydantic already provide use `audit-code`. `AGENTS.md` outranks all three.

## The map

The layers in `pyproject.toml`'s import-linter contract are the design. Each layer owns one kind of decision.

- **physics, model, atomic** own relations with a source. Pure functions of values, no `bpy`, no other seascape.
- **assets, config** own what the user can say and its validation. A pydantic model parses input once; a derived value it exposes is still data, not behaviour.
- **motion, outputs** own pure computation over validated config: where a hull is at a time, what a frame becomes.
- **blender, render** are the imperative shell. They turn values into datablocks, nodes and keyframes. They decide nothing a pure function could decide.
- **server, catalog, zones, sync, rigview, cli** are entry points. They parse, call and print.

Dependencies point inward. A contract violation means the code is in the wrong layer, never that the contract needs an exception.

## Questions per principle

Each principle is a question about this map.

- **Single responsibility (Martin: one actor, one reason to change).** Which decision does this code own? Name the change that would force an edit here. If two unrelated changes would, it holds two decisions. If one change forces edits in several layers (shotgun surgery), a decision leaked out of its home.
- **Information hiding (Parnas).** What design decision does this module hide? A module cut by processing step ("first build the size, then key it", Ousterhout's temporal decomposition) leaks the same knowledge into every step. Cut by the decision likely to change instead.
- **Deep modules (Ousterhout).** Is the interface much smaller than the work behind it? A function whose callers must carry an intermediate value from one call to the next has exposed its internals. Return the finished thing, or take the inputs that make it.
- **Open/closed, read through YAGNI.** Adding a kind of thing should add a member to a discriminated union in one place, not an `if kind ==` in several. Until there is a second kind, there is no extension point to build.
- **Liskov.** Rarely about inheritance here. Ask it of union members and of `Path`-like callables: can every caller use every member without checking which one it got?
- **Interface segregation.** Does a function take a whole scenario or model to read one field? Pass the field. A wide parameter couples the callee to every change in the record.
- **Dependency inversion, the Python way.** The core takes values and plain callables. It never reaches out to Blender or a file. A Protocol or ABC appears at the second implementation, not the first.
- **Functional core, imperative shell (Bernhardt).** Could this branch run on plain values and be tested without Blender? Then it belongs below `blender`. The shell's code has almost no conditionals.
- **Connascence (Page-Jones, Weirich).** How strong, how many, how far apart? Two modules agreeing on a property name, a key order or a formula is strong coupling across a boundary. Strong connascence is fine inside one function and costly across layers. Weaken it to a shared name or type, or move the two halves together.

## Smells this team has hit

Each pattern names its usual root fix.

- **Blender as a message bus.** A value written to a custom property only so another layer reads it back. The reader should receive the value from the computation that made it. Custom properties are for what Blender itself or a user in Blender needs.
- **A closure that rebuilds.** A callback or lambda whose job is to recompute what another layer already computed. Pass the result, or move the computation to the one layer both call.
- **One action, several homes.** Keying an object, sizing a hull, converting a pose: if two call sites do it, a third will drift. One function owns the action; the others call it.
- **The same formula in two modules.** One relation with a source has one home in `physics` or `motion`. Two copies are connascence of algorithm and will disagree after the next fix.
- **A field or default for one caller.** A record field, option or default that exists because one call site wanted it. Compute it at that call site, or ask why the record does not already carry the real quantity.
- **A tag with one legal value.** A `Literal` with one member, or an enum of one, is speculative generality. Delete it until a second value exists; the union arrives with it.
- **Logic in the wrong layer.** Kinematics or integration written out in `config` rather than called from `physics`; a decision (which wave, which leg, which hull) in `blender`. Config validates and derives simple facts; motion computes over time; Blender keys.
- **Feature envy.** A function that reads mostly another module's record. It belongs beside that record.
- **Data clumps and primitive obsession.** The same few floats passed together through several signatures. One `NamedTuple` or frozen dataclass, defined in the lowest layer that uses all of them.

## Choosing the container

- **pydantic model:** untrusted input at the boundary, in `config` and `assets`. `Model` forbids extra keys; `FrozenModel` when nothing may change it after parsing.
- **Discriminated union on a `Literal` tag:** one of several kinds the user picks, registered in one place. Not before the second kind.
- **NamedTuple:** a small immutable value passed between pure functions, built often, read by field.
- **Frozen dataclass:** a trusted in-process record with defaults or methods that a NamedTuple's tuple behaviour would confuse.
- **Plain function or `Callable` type:** behaviour without state. A one-method class is a function.
- **Protocol:** only when two implementations exist and a caller must not know which.

## When principles disagree

| Tension | Settle it by |
|---|---|
| SRP against deep modules | Split on a decision that changes independently. Never split one decision across two shallow modules to make each "do one thing". |
| DRY against the wrong abstraction (Metz) | Merge what must change together: one relation, one action. Leave look-alikes. If a shared helper grew a flag per caller, inline it and look again. |
| OCP against YAGNI | Build the extension point with the second case, in the same change. A hook for an imagined case is inventory. |
| Layer purity against a one-line shortcut | The layer wins. A shortcut across layers is how the smells above start. |
| Moving code against a small diff | Report the misplacement. Move it in its own commit, never inside a behaviour change. |

## Review procedure

1. Read `AGENTS.md`, then the import-linter contracts. Run `uvx --from import-linter lint-imports` first; a failure is the first finding.
2. For the diff, list each new name: function, field, property, parameter, callback, module. Each one must answer "who needs this, and why here?" in one sentence. No answer is a finding.
3. For each new name, find its layer on the map. Ask whether a lower layer could own it, and whether something already does (`grep` the concept, not the spelling).
4. Trace one value end to end: where it is computed, where it is stored, where it is read. Every hop through Blender, a closure or a re-computation needs a reason.
5. Count homes. For each action and each formula the diff touches, count the places that perform it.
6. Ask the principle questions above only where steps 2 to 5 found friction. Do not grade the diff against every principle.
7. Before reporting, find why the code is as it is. Only a documented constraint justifies an odd shape: an `AGENTS.md` trap, a Blender limit or a measured cost. Then it is a keep, with that reason. That the codebase already does it elsewhere is never a reason; it is a second finding.

## What a finding contains

- `file:line` of the symptom, and of the code it duplicates or depends on.
- The smell, named in the words above or by its source (Fowler, Ousterhout, Parnas, connascence).
- Its concrete cost: the next change that will go wrong, or the two places that will drift. "Unclean" is not a cost.
- The root fix and the layer it belongs in. A fix that adds a layer, flag or wrapper is suspect; prefer deleting, moving or passing a value.
- A verdict: **move**, **merge**, **delete**, **keep (reason)** or **open an issue**.

Number the findings. Lead with the one whose fix removes the most coupling. End with what you did not read.

Sources are in [references/sources.md](references/sources.md).

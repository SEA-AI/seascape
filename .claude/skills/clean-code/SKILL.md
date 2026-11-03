---
name: clean-code
description: Write, review or refactor seascape code so a stranger reads it once and changes it safely. Judgment for naming, structure, types, errors, comments, tests and change size, grounded in outside practice rather than local habit. Use before writing a module or function, when a diff feels clumsy, when reviewing your own diff before a commit, or when deciding whether an existing pattern deserves to be copied.
---

# Clean code

Clean code is code whose next change is cheap and safe. Short, clever or familiar code
is none of those by default. This skill gives you questions to ask and a way to settle
disagreements, not a rule list. Stack detail (Python, pydantic, numpy, images, bpy,
pytest) is in [references/stack.md](references/stack.md).

## Order of authority

1. `AGENTS.md`: the project's decisions and the traps that fail silently. Follow it.
2. This skill: the reasoning, for the cases `AGENTS.md` does not cover.
3. The surrounding code: evidence of what the codebase does, not of what is good.

Match the surrounding idiom, except where it is a smell. Do not spread a smell because
it is nearby, and do not fix one in an unrelated diff: open an issue or make it its own
change. A convention earns a place only if you can say what failure it prevents. If it
cannot be said, it is habit.

## Before you write

- Does something already do this: Blender, the standard library, an installed
  dependency, this repo? Reuse beats writing.
- What is the smallest change that solves the stated problem? Name what you leave out.
- Which decision does this code own? If the answer is "several", split it. If a change
  to that decision would touch other files, the decision is leaking.
- What would a caller expect from the name? Does the code do exactly that?
- Which inputs make this an error? Can the types or the semantics rule them out, so no
  caller has to handle them?
- Where does it live? Pure relation with no I/O goes in the innermost layer. A thin
  translation to a file, the CLI or Blender stays at the edge with no logic in it.

## What good looks like

- **Simple means unbraided.** Concerns that can change separately sit in separate
  places. Fewer lines is not the measure; fewer things a reader must hold at once is.
- **Deep modules.** A small interface in front of real work. A wrapper that only renames
  is a cost with no benefit. Cut modules where a decision is likely to change, not where
  the file is long.
- **Parse at the boundary, trust inside.** Turn untrusted input into a type that carries
  the proof, once. Do not re-check the same fact in every layer, and do not pass `dict`
  or `Any` across a function boundary.
- **Make the illegal state unrepresentable.** One record per state beats a record with a
  tag and optional fields. A closed set is an enum or a tagged union.
- **Define errors out of existence, else fail loud and specific.** Every exception is a
  case each caller handles. When one is real, name what failed, the offending value and
  the way out. Never swallow it, never return `None` for it.
- **Names carry intent, in the domain's words.** A good name removes a comment and a
  lookup. A unit or frame belongs in the name or the type.
- **A comment holds what the code cannot:** why, a constraint, a unit, a source, a
  workaround and when it can go. It never narrates the line, the history or the task.
- **Duplication is cheaper than the wrong abstraction.** Merge code that must change
  together (one rule, one formula). Leave look-alikes that may diverge. Abstract on the
  third case. If an abstraction grows flags, inline it and look again.
- **Pure core, thin shell.** Decisions are plain functions of values. I/O and engines sit
  at the edge. Dependencies point inward, and a tool enforces it.
- **Composition over inheritance, function over one-method class.** A Protocol, ABC,
  factory or registry waits for its second implementation.
- **Tests pin behaviour and published relations**, not private names or call order. A
  test pays only for what it checks: smaller input, shared setup, fewer samples.
- **Small, single-purpose changes.** Tidying and behaviour change go in separate
  commits. A commit says why; the diff already says how.

## Where authorities disagree

| Question | Settle it by |
|---|---|
| Tiny functions or longer ones | Split on a nameable concept with a narrow interface, never to hit a line count. One linear flow can stay long. |
| Comment or better name | Name first. Then comment the why, the constraint and the source. A missing why costs more than a stale one, so keep comments few and checkable. |
| DRY or tolerate duplication | Dedupe the decision, tolerate the resemblance. |
| Clean layers or speed | In render and array hot paths, measure before adding indirection; prefer array operations over per-element objects. |
| Lint says split it | A size or complexity limit is a prompt, not an order. Split if the seam is natural. A lint that finds bugs is not negotiable. |

## Smells, as questions

- Does a pure refactor break this test?
- Does a boolean argument change what the function is? Are there two functions in one?
- Is a `str`, `int` or `float` standing in for a closed set, an id or a unit?
- Does the same `if kind == ...` appear in several places?
- Does one change touch many files? Which decision is leaking?
- Does a comment explain what a block does? Would a name or a function say it?
- Is there magic or odd order with no why? What failure does it prevent?
- Who calls this option, hook or base class today?
- Does `try` wrap more than the one call that can fail?
- Can this object be built in a state the code treats as impossible?
- Is this class used by one caller, or is this layer there for a future nobody asked for?

## Before you commit

Read the diff cold, as a reviewer.

- Does it do one thing, and does the title say what?
- Can I delete a line, parameter, option or helper and still pass?
- Does every comment add a reason, unit or source, and avoid the history?
- Does every error path say what failed and what to do?
- Is any hunk a rename or reformat mixed into behaviour? Split it.
- Is there an `except` that swallows, a default that hides a bug, a check that cannot
  fail, debug output, or commented-out code?
- Would the new test fail if the code were wrong?
- Would I approve this diff from someone else?

## Questioning a convention

Before copying a pattern or a rule, ask what established projects do. Read the config of
a few that face the same trade-off. A rule most of them ignore is probably noise; one
they all configure is probably a smell. Keep what survives, and say in the PR what you
checked.

Sources behind this skill are listed at the end of the reference file.

---
name: stack
description: The idiom of each technology in the code: Python, numpy, images, pydantic, bpy and the CLI; not PR stacks, which are `gh-stack`'s. Use when writing code in one of them; `review-pr` runs it as one reviewer. Where code lives is `architecture`'s, whether it should exist `anti-hack`'s, its comments `prose`'s, its tests `tests`'.
---

# Stack

Read the section for the code you are touching. Each practice names the failure it prevents; check the code for that failure. The silent traps specific to this repo are in `AGENTS.md`.

## Python

- **A closed set is an `Enum`, `StrEnum` or `Literal`**, never a string compared with `==`. A typo then fails at the type check.
- **Raise instead of returning `None` for failure.** Callers forget to check, and the failure surfaces somewhere else. Where the absence is normal, say so in the type.
- **Keep `try` to the one call that can fail**, catch what you can handle, and chain with `raise X from err`. `except:` and `except Exception:` hide unrelated bugs.
- **Default arguments are evaluated once.** `None` plus a body check, or `field(default_factory=...)`.
- **A boolean flag that changes what a function does is two functions.** At minimum make it keyword-only so the call site reads.
- **Loop over items, not indices:** `enumerate`, `zip(strict=True)`, `.items()`. Needing an index usually means the structure is wrong.
- **A comprehension stops at one `for` and one filter.** Past that, write the loop or a named helper.
- **A generator beats a list the caller only iterates.**
- **No import-time side effects and no mutable module state.** Build objects in functions; module constants are immutable.
- **`@override` on overrides**, so a renamed parent method fails the checker.
- **Do not golf.** If a walrus, `reduce` or nested expression needs a comment to decode, write the loop. "Practicality beats purity" and "readability counts" are the tiebreak.
- **Lint limits.** `ruff.toml` enforces the counting rules (`C90`, `PL`) too, so CI fails them. Meet a limit by finding the decision that belongs elsewhere, not by splitting one coherent algorithm into one-use helpers, which reads worse than the original.

## Data modeling and pydantic

Pick the container by what the data is:

| Data | Use |
|---|---|
| Untrusted input (TOML, JSON, CLI), or config that needs validation | the repo's pydantic base model, which forbids extra keys, or its frozen variant |
| Trusted in-process record | `@dataclass(frozen=True)` |
| Small fixed record unpacked by position | `NamedTuple` |
| A dict shape you must keep as a dict | `TypedDict` |
| One value with meaning or rules | `NewType`, or a value object if it has behaviour |
| Behaviour without data | a function |

- **Make the illegal state unrepresentable.** One record per state beats a record with a tag and optional fields.
- **Put each invariant in one place**: the validator or constructor. Scattered `if` checks get missed.
- **Forbid unknown keys at config boundaries**, or a misspelled key is ignored in silence.
- **Know what skips validation:** `model_construct` and `model_copy(update=...)`. To change a model safely, validate a modified dict. Anything that relies on a validator's derived fields breaks otherwise.
- **Raise `ValueError` in a validator, never `assert`**, which `-O` removes.
- **Choose strict or lax coercion on purpose.** Lax turns `"123"` into `123`.
- **Accept abstract types (`Mapping`, `Sequence`, `Iterable`), return concrete ones.** Contain `Any`; `object` means "accepts anything".
- **Immutable by default** for config. `frozen` guards the attribute, not a dict it holds.

## Numeric and image code

- **State shape, dtype and frame** of each array argument in the docstring: `(N, 3)`, `float64`, row-major. The annotation does not carry it.
- **Randomness:** never `seed + i`; record the seed with the output.
- **Rewrite formulas that lose precision.** `log1p` and `expm1` keep the digits `log(1 + x)` and `exp(x) - 1` drop for a small `x`; `hypot` avoids overflow in `sqrt(x*x + y*y)`; `arctan2` keeps the quadrant `arctan(y / x)` loses.
- **Be explicit about dtype** at creation and at I/O. Under NEP 50 the array dtype wins over a Python scalar, so `uint8 + 200` wraps with no warning. OpenCV's `add` saturates and NumPy wraps, so mixing them gives two answers.
- **Decide view or copy.** Basic slices return views, and `reshape` does where it can, so a write mutates the caller's array. `.copy()` where a function must not.
- **NaN and inf are signals.** Wrap code that should never produce them in `np.errstate(invalid="raise")`. Integer overflow in an array is never flagged.
- **Do arithmetic in linear radiance**; store gamma-encoded values only at the file edge. Averaging or resizing encoded values blends wrong.
- **Axis order:** arrays are `(row, col, channel)`; Pillow `size` and OpenCV `dsize` are `(width, height)`. OpenCV reads BGR, Pillow RGB. A swap still renders plausibly.
- **Resizing a label or index image with a smoothing filter invents classes.** Use nearest.
- **Vectorise over independent data axes; loop over axes that carry state or are tiny.** A broadcast that needs more than two or three `newaxis` tricks is harder to read than the loop. Write the loop first, and vectorise when it is measurably slow.

## Blender and bpy

- **Write through `bpy.data`; use `bpy.ops` only when there is no data API.** Operators take context, not arguments, and fail with a bare `RuntimeError`. If one is needed, supply context with `Context.temp_override()`.
- **Keep the object a creator returns and do not look it up by name again.** A requested name gets a `.001` suffix when taken, and a later lookup finds the wrong datablock.
- **Do not hold bpy references across undo, mode changes or adding and removing in the same collection.** Re-fetch or keep them inside one function. Copy a collection to a list before mutating it in a loop.
- **One function per sub-graph:** it takes the tree and input sockets and returns the output socket. Raw `links.new` calls live in a few helpers, which also set defaults you depend on explicitly.
- **Properties with order dependence are set in the documented order**, since the wrong order is accepted without error. `AGENTS.md`'s traps list the known ones.
- **Bulk data goes through `foreach_set` and `foreach_get` on flat numpy arrays**, not per-item loops.

## CLI

- **A CLI command writes data to stdout**, diagnostics to stderr, and exits 0 on success.

## Errors

- **Error message:** what failed, the offending value, the valid range or fix. One problem per message: `f"radius_m must be > 0, got {radius_m!r}"`.
- **`logging` for events, `print` only for CLI output.** Libraries do not configure the root logger.

## What a finding contains

Report a table, the likeliest failure first:

| # | Where | Verdict | Finding | Fix |
|---|---|---|---|---|
| 1 | `file:line` | fix | a mutable default is shared across calls, so the list grows | default `None`, built in the body |

- **Verdict:** **fix** or **keep**. A keep gives its reason in Fix.
- **Finding:** the practice broken and the failure it causes.

End with what you did not read.

Sources are in [references/sources.md](references/sources.md).

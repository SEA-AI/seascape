---
name: stack
description: The practice of each part of seascape's stack: Python, numpy, images, pydantic, bpy, the CLI and tests. Use when writing code in one of them, for its idiom; `review-pr` runs it as one reviewer. Where code lives is `design`'s, whether it should exist `anti-hack`'s, its comments `prose`'s.
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
- **`match` destructures; a plain `if` tests.** Use it where it reads better.
- **No import-time side effects and no mutable module state.** Build objects in functions; module constants are immutable.
- **`@override` on overrides**, so a renamed parent method fails the checker.
- **Do not golf.** If a walrus, `reduce` or nested expression needs a comment to decode, write the loop. "Practicality beats purity" and "readability counts" are the tiebreak.
- **Lint limits.** Rules that detect bugs (`B`, `F`, undefined names, `raise` without `from`) are obeyed. Rules that count (arguments, branches, statements, complexity, magic numbers) are prompts: pandas ignores them and fastapi ignores `C901`. One coherent algorithm split into five one-use helpers reads worse than the original.

## Data modeling and pydantic

Pick the container by what the data is:

| Data | Use |
|---|---|
| Untrusted input (TOML, JSON, CLI), or config that needs validation | pydantic `BaseModel`, `frozen=True`, `extra="forbid"` |
| Trusted in-process record | `@dataclass(frozen=True, slots=True, kw_only=True)` |
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
- **Randomness:** never `seed + i`; record the seed with the output. A seeded stream is not stable across NumPy releases, so test statistics and properties, not exact draws.
- **Rewrite formulas that subtract nearby values.** `log1p`, `expm1`, `hypot` and `arctan2` exist for this; `1 + x` drops the digits of a small `x`.
- **Be explicit about dtype** at creation and at I/O. Under NEP 50 the array dtype wins over a Python scalar, so `uint8 + 200` wraps with no warning. OpenCV's `add` saturates and NumPy wraps, so mixing them gives two answers.
- **Decide view or copy.** Basic slices and `reshape` return views, so a write mutates the caller's array. `.copy()` where a function must not.
- **NaN and inf are signals.** Wrap code that should never produce them in `np.errstate(invalid="raise")`. Integer overflow is never flagged.
- **Do arithmetic in linear radiance**; store gamma-encoded values only at the file edge. Averaging or resizing encoded values blends wrong.
- **Axis order:** arrays are `(row, col, channel)`; Pillow `size` and OpenCV `dsize` are `(width, height)`. OpenCV reads BGR, Pillow RGB. A swap still renders plausibly.
- **Resizing a label or index image with a smoothing filter invents classes.** Use nearest.
- **Vectorise over independent data axes; loop over axes that carry state or are tiny.** A broadcast that needs more than two or three `newaxis` tricks is harder to read than the loop. Write the loop first, vectorise when it is measurably slow, and keep the loop as the test oracle.

## Blender and bpy

- **Write through `bpy.data`; use `bpy.ops` only when there is no data API.** Operators take context, not arguments, and fail with a bare `RuntimeError`. If one is needed, supply context with `Context.temp_override()`.
- **Keep the object a creator returns and do not look it up by name again.** A requested name gets a `.001` suffix when taken, and a later lookup finds the wrong datablock.
- **Do not hold bpy references across undo, mode changes or adding and removing in the same collection.** Re-fetch or keep them inside one function. Copy a collection to a list before mutating it in a loop.
- **One function per sub-graph:** it takes the tree and input sockets and returns the output socket. Raw `links.new` calls live in a few helpers, which also set defaults you depend on explicitly. Name and label every node by intent, use a Frame per stage, and put the layout rule in one place.
- **Isolate every version-dependent name behind one small adapter** or a table keyed on `bpy.app.version`, so a Blender release breaks one test, not the build.
- **Properties with order dependence are set in the documented order**, since the wrong order is accepted without error.
- **Bulk data goes through `foreach_set` and `foreach_get` on flat numpy arrays**, not per-item loops.
- **Test cheaply:** the pure layer needs no Blender. Assert graph structure on a fresh scene. Evaluate a graph with one small probe render per property and compare with the pure function's value. Share one built scene across tests.

## CLI and tests

- **A CLI command writes data to stdout**, diagnostics to stderr, and exits 0 on success.
- **A test never restates config.** Asserting that a default, a preset or a TOML field equals the literal written there checks nothing and changes with it. Test what the value does.
- **A test fails when the code is wrong.** A check that cannot fail is dead. Propose the smallest test that pins the behaviour.
- **Test what the code promises:** outputs, visible state, an invariant or a published relation with its source. Do not assert private names, call order, mock arguments, log wording or tuned numbers.
- **One reason to fail per test**, a name that states the behaviour, `parametrize` for a table of cases. DAMP beats DRY in tests: a test reads top to bottom.
- **Prefer real collaborators.** Double only what is slow, nondeterministic or foreign, and wrap a third-party API in a thin facade you own rather than mocking its objects.
- **Test a physical formula** by analytic limits, invariants (symmetry, monotonicity, scaling), and an independent reference. Justify every tolerance in a comment and set `atol` explicitly for values near zero. Golden images only on tiny fixtures with a tolerance.
- **A helper, fixture or `conftest` layer used by fewer than three tests** is not yet worth its indirection.

## Errors

- **Error message:** what failed, the offending value, the valid range or fix. One problem per message: `f"radius_m must be > 0, got {radius_m!r}"`.
- **`logging` for events, `print` only for CLI output.** Libraries do not configure the root logger.

## Sources

- Google engineering practices: https://google.github.io/eng-practices/
- Google Python style guide: https://google.github.io/styleguide/pyguide.html
- PEP 20, 257, 387, 698: https://peps.python.org/
- King, parse don't validate: https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/
- pydantic validation docs: https://pydantic.dev/docs/validation/latest/
- Typing best practices: https://typing.python.org/en/latest/reference/best_practices.html
- Hynek, subclassing redux: https://hynek.me/articles/python-subclassing-redux/
- Hynek, what to mock: https://hynek.me/articles/what-to-mock-in-5-mins/
- Fowler, mocks aren't stubs: https://martinfowler.com/articles/mocksArentStubs.html
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- clig.dev: https://clig.dev/
- NumPy: parallel RNG, NEP 19, NEP 50, copies and views, `errstate`, `isclose`
- Goldberg, floating point: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
- Wilson et al., best practices for scientific computing: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745
- Blender Python API gotchas and the 4.0 and 5.0 Python release notes: https://docs.blender.org/api/current/info_gotcha.html
- Ruff configs of pandas, fastapi, pydantic and httpx, read from each `pyproject.toml`

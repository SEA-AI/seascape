# Sources

## Published

| Rule | Source |
|---|---|
| Narrow `try`, no catch-all, `raise from` | Google Python style guide §2.4: https://google.github.io/styleguide/pyguide.html; ruff `B904` |
| Defaults are evaluated once | Python tutorial §4.9.1: https://docs.python.org/3/tutorial/controlflow.html#default-argument-values |
| A boolean flag is two functions | Fowler, flag argument: https://martinfowler.com/bliki/FlagArgument.html |
| Loop over items; `zip(strict=True)` | Google §2.8; PEP 618 |
| A comprehension stops at one `for` and one filter | Google §2.7 |
| No mutable module state | Google §2.5 |
| `@override` | PEP 698 |
| Do not golf | PEP 20 |
| Behaviour without data is a function | Hynek, subclassing redux: https://hynek.me/articles/python-subclassing-redux/ |
| Illegal state unrepresentable; each invariant in one place | King, parse don't validate: https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ |
| Forbid unknown keys; what skips validation; `ValueError` not `assert`; coercion; faux immutability | pydantic models and validators: https://docs.pydantic.dev/latest/concepts/models/ |
| Abstract arguments, concrete returns; `Any` versus `object` | https://typing.python.org/en/latest/reference/best_practices.html |
| Never `seed + i`; streams may change | NumPy parallel RNG: https://numpy.org/doc/stable/reference/random/parallel.html; compatibility policy: https://numpy.org/doc/stable/reference/random/compatibility.html |
| Record the seed; optimise only after it works | Wilson et al. 2014: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745 |
| `log1p`, `expm1`, `hypot`, `atan2` | https://docs.python.org/3/library/math.html |
| NaN and inf as signals | `numpy.errstate`: https://numpy.org/doc/stable/reference/generated/numpy.errstate.html |
| OpenCV's `add` saturates | OpenCV, arithmetic operations on images: https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html |
| Axis order; BGR versus RGB | Pillow `Image.size`: https://pillow.readthedocs.io/en/stable/reference/Image.html; `cv2.resize` and `cv2.imread` |
| `uint8 + 200` wraps | NEP 50 |
| Views and copies | https://numpy.org/doc/stable/user/basics.copies.html |
| Operators take context and raise `RuntimeError` | Blender API, using operators: https://docs.blender.org/api/current/info_gotchas_operators.html |
| `temp_override` | https://docs.blender.org/api/current/bpy.types.Context.html#bpy.types.Context.temp_override |
| Keep the created object | Blender API, internal data: https://docs.blender.org/api/current/info_gotchas_internal_data_and_python_objects.html |
| No references across undo, mode change, reallocation | Blender API, crashes: https://docs.blender.org/api/current/info_gotchas_crashes.html |
| stdout for data, stderr for diagnostics | https://clig.dev/#the-basics |
| `logging` for events; libraries leave the root logger | Python logging HOWTO: https://docs.python.org/3/howto/logging.html |
| An error message names the condition and the fix | Google §3.10.2; https://clig.dev/#errors |

## Repo decisions

- Silent traps live in `AGENTS.md`; order-dependent properties are listed there.
- `ruff.toml` enforces the counting rules.
- The pydantic base models in `seascape/model.py`.
- The findings table, shared by every reviewer.

## Judgement

Widely held, with no single source:

- A closed set is an `Enum`, `StrEnum` or `Literal`.
- Raise instead of returning `None`.
- A generator beats a list the caller only iterates.
- Import-time side effects; one problem per error message.
- The container table's `dataclass`, `NamedTuple`, `TypedDict` and `NewType` rows.
- State shape, dtype and frame in the docstring.
- Arithmetic in linear radiance; nearest for label images.
- Prefer `bpy.data` to `bpy.ops`.
- Vectorise data axes, loop stateful ones.
- One function per sub-graph; `foreach_set` for bulk data.

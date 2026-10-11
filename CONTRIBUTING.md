# Contributing

[`AGENTS.md`](AGENTS.md) covers the conventions and the Blender traps to know before changing anything.

Only touching the radiometry or the scenario config? `uv sync --no-group blender` skips the Blender wheel, hundreds of MB. Those parts are plain NumPy and run without it.

Install the hooks once, before your first commit:

```bash
uvx pre-commit install
```

That runs `ruff check --fix` and `ruff format` on what you staged. The rest of CI is these commands, all of which have to pass:

```bash
uvx ruff check .
uvx ruff format --check .
uvx ty check
uv run pytest
```

The render checks are the only thing that catches a sea or sky shader rendering wrong. Cycles renders them on a GPU where there is one and on the CPU otherwise, as on CI:

| command | runs |
|---|---|
| `uv run pytest` | the suite and the render checks |
| `uv run pytest -m "not render"` | the suite, without the render checks, while iterating |
| `uv run pytest -m render` | the render checks alone |
| `uv run pytest -n 0` | in one process, for `--pdb` or a print |

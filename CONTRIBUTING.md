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

The render checks are the only thing that catches a sea or sky shader rendering wrong. They need a GPU, CI never runs them, and they belong before any change to that chain:

| command | runs |
|---|---|
| `uv run pytest` | the suite, without the render checks, as CI does |
| `uv run pytest --render` | the suite and the render checks |
| `uv run pytest --render -m render` | the render checks alone |

# Assets

`seascape assets` lists the meshes a scenario can name in `asset`, and the skies it can name in `sky.hdri`.

![Every mesh, side on and three-quarter on](assets.jpg)

## Where a mesh may come from

A build downloads each mesh itself, so it needs a URL that anyone can fetch without logging in and that always returns the same bytes.

| Source | Licence | URL |
|---|---|---|
| [Poly Haven](https://polyhaven.com/models) | CC0 | `dl.polyhaven.org`, from `api.polyhaven.com/files/<id>` |
| [Objaverse](https://huggingface.co/datasets/allenai/objaverse) | Each model's own: take CC0 or CC-BY only | `resolve/<commit>/glbs/...`, pinned to a commit, never `main`; `object-paths.json.gz` maps a Sketchfab uid to its path |
| [Icosa Gallery](https://icosa.gallery) | CC-BY | `api.icosa.gallery/v1/assets/<id>`; skip `CREATIVE_COMMONS_BY_ND` |

Not usable:

- A non-commercial or no-derivatives licence.
- Sketchfab directly: its Download API needs an account and hands out links that expire.

## Adding one

1. Find it in a source above and read its licence.
2. Add an entry to [`seascape/assets.toml`](../seascape/assets.toml): `url`, `sha256` (`shasum -a 256` of the file), `licence`, and `attribution` with the author and the page.
3. Set `length_m` and `draught_m` from a real vessel of the class.
4. Set `bow_deg` to where the bow points as authored, and check it in the sheet.
5. Fill `triangles` and `texture_px` from `uv run python -c "from seascape import scene; print(scene.measure('<name>'))"`, and write a one-line `description` of what the camera sees.
6. Regenerate the sheet with `uv run python docs/assets.py docs/assets.jpg`, and run `uv run pytest --render`.

## Judging detail

A shape or texture smaller than a pixel cannot be seen. A camera whose pixel spans `ifov` radians sees a detail of `d` metres only while the range is under `d / ifov`. Lindstrom & Pascucci, "Visualization of Large Terrains Made Easy", IEEE Visualization 2001, Eq. 2, select terrain detail this way. A low-poly hull suits targets at range; an ownship or a close target needs textures and fine geometry.

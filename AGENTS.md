# AGENTS.md

What an agent needs that a human picks up by working here. Everything else — what the project is, how to install it, how to run it — lives in [README.md](README.md).

## The rule

**Blender does anything it can do. Code is a liability.**

Before writing a module, check whether Blender already has the feature. The sky is the Sky Texture. Depth and segmentation are render passes.

Two things are **not** Blender's, both documented so nobody helpfully puts them back.

**Waves are shading normals, not the Ocean modifier.** Displaced geometry goes sub-pixel before the horizon, and sub-pixel geometry aliases instead of averaging. Each pixel draws the waves longer than its footprint and takes the rest as roughness; in EO, part of the rest is a random tilt per cell of one specular point, so the sun glitters. The sea does carry geometry, for the earth's curve -- kilometres across, never sub-pixel. The accepted cost is that a normal cannot occlude, so a wave can never hide a target. Run `pytest --render` before touching the sea shader.

**LWIR radiometry lives in numpy.** Blender has no concept of an 8–14 µm band, and its Fresnel node takes a scalar IOR where seawater emissivity needs complex IOR (n + i·k).

If a new dependency looks necessary, say why Blender or the standard library can't do it.

## The dev loop

**Code never runs inside Blender.** `seascape` builds a `.blend` in its own process; Blender opens it. The dependency between them is a file, not an import — so there is nothing to hot-reload and no module cache to defeat.

Do not add `sys.path` entries, `.pth` files, `importlib.reload`, symlinks into Blender's script directories, or an add-on wrapper to get around this. Every one of them exists to let Blender host your code, which is the thing to avoid.

The loop is `seascape build`, then reload in Blender.

To reload without losing where the user had the viewport, via MCP:

```python
import bpy
from mathutils import Matrix


def _view3d():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                return next(s.region_3d for s in a.spaces if s.type == "VIEW_3D")


path = bpy.data.filepath  # or the .blend seascape just wrote
rv = _view3d()
view = Matrix(rv.view_matrix), rv.view_distance, rv.view_location.copy()
bpy.ops.wm.open_mainfile(filepath=path)
rv = _view3d()  # regions are rebuilt by open_mainfile
rv.view_matrix, rv.view_distance, rv.view_location = view
rv.update()
```

Without MCP, **File → Revert** (`bpy.ops.wm.revert_mainfile`) does the same minus the view restore. Either way, reloading discards unsaved in-memory changes, so anything hand-tweaked in Blender is lost — put it in the build code instead.

Building fresh each time is also what keeps a long-lived session from accumulating state that the next build inherits.

## Traps that fail silently

These produce wrong output with no error. They are the reason this file exists.

- **A bearing is negated only when it becomes a rotation.** Blender's +Z rotation turns a forward-facing object to port, so `blend.yaw` negates, and nothing else does. A direction is `(sin, cos)` of the bearing and `atan2(x, y)` reads one back; neither negates. The Sky Texture's `sun_rotation` is a direction, whatever its name says.
- **Wind and swell are named for where they come from; waves run the other way.** `wind_from_deg` and `swell.from_deg` are bearings of origin, and `waves` turns them by 180 once. A second turn looks right in a still.
- **A camera's bearing is not `pod yaw + camera yaw`.** In the chain `scene._rig` builds, the rig's pitch sits between the two yaws and shifts an off-axis camera's azimuth. A centre camera on a level hull is exact, which is why the sum looks right. `scene.boresight_deg` reads the achieved bearing off `matrix_world`; `Mount.nominal_bearing_deg` is only what the scenario asked for.
- **A glTF import sets `rotation_mode` to `QUATERNION`.** Assigning `rotation_euler` is then ignored entirely — no exception, no warning, object doesn't move. Set the mode first.
- **A slope sigma is total or per axis.** Cox & Munk's is total; a Beckmann alpha, which GGX borrows, is sqrt(2) sigma per axis; `lwir.emissivity_curve` takes sigma per axis. Swapping them is off by sqrt(2) and renders plausibly.
- **The glitter is tuned to `outputs.samples.eo`.** Each cell's lobe widens so that many samples count its glints right. Render the `.blend` at another sample count and it sparkles more or less, silently.
- **`model_copy` skips validation, and validation is what gives the air and sea their atmosphere's temperatures.** Swap `sky.atmosphere` with `model_copy` and the old profile's air and sea stay. Validate a dict instead.
- **A Math node's unlinked inputs default to 0.5.** A `MULTIPLY_ADD` chain picks up 0.5 from the first addend unless it is set to 0. Vector Math defaults to 0. Its `DOT_PRODUCT` and `LENGTH` answer on `Value`; their `Vector` output is disabled, so `outputs["Vector"]` raises and `outputs[0]` reads 0.
- **Address shader sockets by name, never by index.** `inputs["Distance"]` raises if Blender renames it; `inputs[1]` happily writes to whatever now sits in that slot.
- **Objects can share a mesh datablock.** Material slots link to mesh data by default, so assigning a material to one object silently changes the other. Use `slot.link = "OBJECT"` when they must differ.
- **LWIR waves do not need `t_sea_k - t_air_k`.** A tilted facet reflects a different elevation of a sky that runs cold overhead to ambient at the horizon, so relief shows with the sea exactly at air temperature. Under a uniform ambient world it does not.
- **The engine identifier is version-dependent.** `BLENDER_EEVEE` means EEVEE Legacy on ≤4.1 and EEVEE Next on ≥5.0, with `BLENDER_EEVEE_NEXT` in between. The Sky Texture's `NISHITA` is `SINGLE_SCATTERING` on ≥5.0; `MULTIPLE_SCATTERING` is a new model.
- **Do not validate an engine against the enum.** Under the `bpy` module `render.engine` reports only `['BLENDER_EEVEE']`, on the class and the instance alike, because Cycles registers as an add-on. Assigning `CYCLES` works anyway and reads back. Assign it and let Blender raise: an identifier it does not know is a `TypeError`.
- **Both clip planes cut in silence.** `clip_end` defaults to 1000 m, so a target at 2 km renders as sky and the boundary reads as a convincing horizon. `rig.near_clip_m` does the same to anything nearer than itself, and Blender accepts a near plane beyond the far one by rendering an empty frame. Nothing warns at either end.
- **`matrix_world` is stale until the depsgraph runs.** Parent an object, move the parent, read a child's `matrix_world`, and you get where it used to be. `view_layer.update()` first, or every measurement quietly describes the wrong scene.
- **An empty's `bound_box` is a point at its origin.** An import is largely empties, so measuring the extent of "everything I just imported" inflates it and the fit comes out wrong.
- **Cycles denoising is on by default and is not radiometric.** OIDN is an edge-aware image filter. On a world flat at 290.00 K it returns 282.43-293.00 K, worst at the frame border, and it breaks the R=G=B that an LWIR scene guarantees. It is off in both bands: in EO it also smooths the sea's waves.
- **An image's `colorspace_settings` must be set before its pixels, never after.** Assigning it second re-reads the buffer that is already there and leaves the image black, with no error.
- **Blender's GPU compositor keeps memory from every render**, about 65 MB a 1080p frame even through a tree that only passes the render on. A clip dies with SIGKILL and nothing in the log. `build` puts the compositor on the CPU.
- **The compositor is the camera, and the build picks it by format.** An 8-bit EO build puts glare, blur and auto-exposure there; an exr build passes the radiance through. Switch the file format after `build` and an exr holds the look, a png the raw frame.
- **`view_settings.exposure` is part of the display transform.** A png carries it, a float EXR ignores it. Same scene, same knob, two formats, and nothing reports the difference.
- **EEVEE caps reflected world light at 10.** `world.sun_threshold` moves anything
  brighter into a sun a mirror cannot see, and `eevee.clamp_surface_indirect` clips
  what is left; a mirror reflects a world of 40 only with both at 0. LWIR radiance is
  tens of W m^-2 sr^-1, so the capped sea reads cold. Unclamped, EEVEE still darkens
  rough reflections at grazing view: Cycles for anything a pixel value is read from.
- **`refresh_devices()` is what actually enables the GPU.** Without it
  `compute_device_type` and `scene.cycles.device` leave Cycles on the CPU, silently, at
  about the same speed. `denoising_use_gpu` changes nothing measurable at 4K. Configure
  the device once before the first render: switching mid-process pays Metal kernel
  compilation and the render comes out three times slower.
- **An FBX import sets the scene's fps to the file's own**, even with `use_anim=False`, and a fractional rate lands in `fps_base`: 29.97 is fps 30 over 1.001. Anything that sets the timeline has to run after the last import, and set both.
- **The sequencer tones its strips through the scene's view transform.** The factory AgX takes a png's white to about 195. `Standard` passes a picture through.
- **A missing strip frame encodes as black**, in silence. Check the files exist first.
- **A movie `render.filepath` without an extension gets the frame range appended**: `cam` is written as `cam0001-0300.mp4`.
- **A Sky Texture with its sun disc on ignores its Vector input.** The socket is disabled, so a link to it stays in the tree and changes nothing, and `inputs["Vector"]` raises. Turn the disc off to point the node anywhere but the ray.
- **The Sky Texture's `turbidity` does nothing under the scattering models.** It belongs to Preetham and Hosek-Wilkie. Haze there is `aerosol_density`. Setting the wrong one is accepted in silence and changes no pixel, which was verified by rendering both.

## Conventions

- **Degrees at the boundary, radians inside.** Config and ground truth are `*_deg`; internals are radians, converted exactly once. Degrees-versus-radians is the live bug class here.
- **Every physical number cites a source**, in a comment beside it: a published relation, or a derivation from one, or a measurement of Blender itself that a test pins. Never a constant fitted to a render. Renders of an eyeballed scene are not a physics target, and a fitted number cannot be checked by anyone reading the diff.
- **Units in field names.** `height_m`, `t_sea_k`, `range_m`. No units library.
- **Randomness comes from named substreams** off the scenario seed — `_substream(seed, "sea/surface")`. Never `np.random` module functions. Named streams mean adding a component doesn't perturb an existing one.
- **Poses are functions of time.** Ground truth records a timestamp, not a frame index: sensors run at different rates, so frame *n* is not one instant.
- **Never commit a `.blend` or an asset.** `.blend` files are build artifacts; assets are fetched from `assets.toml` into a cache outside the repo.
- **Components are pydantic models; a second kind of one makes a discriminated union**, registered in one place. Not a registry dict, not entry points.
- Annotate every function signature outside tests.

## Where code goes

- **Physics imports no `bpy`.** A relation with a source lives in a module tested without Blender; Blender-side code only turns its values into nodes and keyframes, and its tests check the graph carries them without re-deriving.
- **One wave field**, from `scene.wind_waves` and `scene.swell_waves`.
- **Pass a `np.random.Generator`, never a seed;** the caller names the substream.
- **No Protocol or ABC before a second implementation.**

## Checks — run before committing

These mirror CI; all must pass.

```bash
uvx ruff check .
uvx ruff format --check .
uvx ty check
uv run pytest
```

ruff and ty go through `uvx` deliberately: never add them to a dependency group and never pin them, `.pre-commit-config.yaml` included. `uvx pre-commit install` automates the ruff lines.

The render-drift check runs only with `--render` and needs a GPU to finish in reasonable time, which is also why CI never runs it. Run it locally before touching anything in the shader chain.

## Working here

- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `ci:`).
- **Branches:** matching prefixes (`feat/...`, `fix/...`).
- **Be concise.** Commit messages, PR descriptions, comments and docs state the fact, not the journey. No debugging narration, no restating the diff, no closing paragraph that repeats what was just said.
- **Comments are for what the code cannot say.** A better name beats a comment explaining a worse one. Write one for a non-obvious constraint, a unit, a workaround, a source — never to restate the line. `# negate the bearing` is noise; `# Blender's +Z turns to port` is the reason.
- **Nothing that can go stale, nothing from outside the code.** No numbers derived from values elsewhere, no lists or values copied from another file, no names of tests, no PRs, reviews, conversations or history, no hardware, specs or consumers the repo does not model. If a claim matters, a test asserts it; the comment does not point there. When in doubt, delete the line.
- **Avoid the machine cadence.** `X, not just Y` for emphasis, lists of exactly three, uniformly long sentences, hedges like "it's worth noting". Prefer the specific: a number, a file name or a flag beats an adjective.
- **Settle a tooling or convention question by what established projects do.** Read the config of a few popular projects facing the same trade-off, and cite them in the PR. A lint rule most of them ignore is likely noise; one they configure rather than ignore is likely a smell.
- **Make PRs scannable.** A table, a before/after, or a rendered frame beats a paragraph.
- Renders are cheap and settle arguments. If a change affects what the camera sees, show it.

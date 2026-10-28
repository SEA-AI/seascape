---
name: label-zones
description: Give an asset's material slots or islands their LWIR thermal zones in seascape/assets.toml, by reading the sheets `seascape zones` renders, and render the contact sheet a person approves. Use when an asset in assets.toml has no `zones` table, or a new asset is added.
---

# Label thermal zones

An LWIR build needs to know what each part of a mesh is made of. You read numbered
renders of the asset and write one zone per part into `seascape/assets.toml`. A person
approves the result from a contact sheet.

## The vocabulary

Exactly one of these per part, as `seascape.physics.thermal.ZoneName` lists them:

| zone | what |
|---|---|
| `paint_hull` | painted shell plating, topsides and bottom |
| `paint_superstructure` | painted house, bridge, masts, cranes, hatch covers, buoy bodies |
| `deck` | walking surfaces: steel decks, teak, non-slip |
| `glass` | windows, portholes, windscreens, lamp lenses |
| `bare_metal` | unpainted steel, stainless, aluminium, chrome |
| `rubber` | inflatable tubes, fenders, tyres, hoses |
| `fabric` | canvas, biminis, cushions, sails, rope |
| `funnel` | the stack's casing |
| `exhaust` | the hot outlet inside or atop a funnel, an exhaust pipe |
| `engine_cowl` | outboard engine covers |
| `wood_composite` | timber, plywood, logs, GRP/carbon hulls of small craft |

The waterline is not a zone: it comes from `draught_m`.

## Steps

1. Read the asset's line in `seascape/assets.toml`, or `uv run seascape assets show
   <asset>`: its `description` says what the vessel is.
2. `uv run seascape zones <asset> -o <dir>`. An asset of more than one material gives
   slot sheets; one of a single material gives island sheets. `--islands` forces
   islands, to split a slot of two zones; `--material NAME` marks only that material's
   islands, the rest grey. The output directory must be empty.
3. Read `<dir>/slots.json` (or `islands.json`). Each part has its material, objects,
   triangles, colour (flat or mean texel), alpha, metallic, and `height_m`: its lowest
   and highest point above the waterline. Alpha is a constant, or a texture's stats
   (`mean`, `min`, `over_half`). An untextured material's islands take its flat base
   colour. A textured island's colour is the mean texel at its triangles' centres,
   which mirrored halves and shared UVs share.
4. Look at `<dir>/overview.png`: the textured views, then every part in its own colour
   with its number. A slots overview ends in a legend. Then open `slot-N.png` (or
   `island-N.png`) for each part: it is bright orange, the rest grey, cropped to it in
   every view that shows it. An island's sheet exists only when it is large enough to
   mark; `atlas-*.png` shows every island over the texture it reads.
5. Decide each part's zone, by these rules:
   - A constant alpha below 1 is glass. An alpha texture is a cutout, a
     railing or a mesh drawn on a plane, not glass. Glass with alpha 1 also exists: a
     dark, flat band of windows on a house is glass. Windows painted into a texture
     share their island with the wall and stay the wall's zone.
   - Never invent a funnel on a yacht, a sailboat, a RIB or a buoy. `funnel` and
     `exhaust` need a ship with a stack.
   - A small part, a rail, antenna, cleat or ladder, is `bare_metal` only when it is
     clearly metal: grey or chrome, thin, and not painted the colour of what carries it.
     Otherwise it takes the zone around it.
   - Below the waterline and on the topsides is `paint_hull` for a painted ship;
     a small craft's moulded hull is `wood_composite`.
   - Metallic maps are unreliable: use them to break a tie, never against the picture.
   - A part that holds two zones: give its material the zone of most of its area, then
     run `--islands --material NAME` and override the islands that differ.
     An atlas of thousands of islands is labelled the same way: the material takes the
     majority zone and only the large marked islands are overridden.
   - When unsure, report the part and why.
6. Write the result (below), then render the contact sheet:
   `uv run seascape zones <asset> -o <dir> --zones` writes `<dir>/zones.png`. Look at it:
   every surface in the zone it should be, nothing in a zone it should not. Hand it to a
   person to approve.
7. `uv run pytest tests/test_zones.py` checks every material is named and the islands
   are numbered as the import numbers them.

## Writing assets.toml

Under the asset, after its other keys:

```toml
[zodiac.zones.materials]
Carbon_Fiber = "wood_composite"
Glass = "glass"
Rubber = "rubber"
```

Every material of the import needs a zone, by its name as `slots.json` gives it; quote
a name that is not a bare key (`"01_-_Default" = "paint_superstructure"`).

Islands override their material's zone, by id. They need `island_count` and
`island_digest` from `islands.json`: the import's number of islands and a fingerprint of
their numbering, so an import that renumbers its islands fails the tests instead of
moving the zones:

```toml
[yacht.zones]
island_count = 1082
island_digest = "a6c3a5bae7ac1fbe"

[yacht.zones.materials]
"01_-_Default" = "paint_superstructure"

[yacht.zones.islands]
"0" = "paint_hull"
"5" = "deck"
```

An island not listed takes its material's zone. Run `uvx tombi format` after editing.

---
name: label-zones
description: Give an asset's parts their LWIR thermal zones in seascape/assets/assets.toml, from the sheets `seascape zones` renders, and render the contact sheet a person approves. Use when an asset has no `zones` table, a new asset is added, or a zone sheet shows a part wrong.
---

# Label thermal zones

An LWIR build gives every face of an asset the emissivity, absorptance and interior of
its zone. You decide what each part of the mesh is made of, write it into
`seascape/assets/assets.toml`, and prove it with a sheet a person approves.

## Where the facts live

Read these:

- `seascape.physics.thermal`: `ZoneName` is the vocabulary and what each zone covers;
  `ZONES` gives each zone's numbers and the source behind them.
- `seascape.assets.Zones` and `Rule`: what a zones table may hold, and how materials,
  rules and listed islands combine. Their validators say what is refused.
- `uv run seascape zones --help`: the sheets it renders and how to aim them.
- `slots.json` or `islands.json` beside the sheets: each part's material, colour,
  inputs and height, measured from the import. Island means are what a rule reads.
- The zones tables already in `assets.toml`: working examples of every form.

## What a zone means

A zone is what the surface emits and absorbs, so the top layer decides it. Paint on
steel emits as paint; bare stainless reflects the sky and reads cold. A small craft's
moulded hull is not a ship's painted plating. The waterline is no zone: the sea meets
the hull at the asset's `draught_m`, so zone the hull below it as above. When two
zones fit, pick the one whose `ZONES` numbers fit this surface, and weigh how much an
error would show: a part a few pixels wide matters less than a hull side.

## Weighing the evidence

Trust the evidence in this order:

1. The textured render: colour, sheen, shape, and what the part is attached to.
2. Context: the asset's `description`, where the part sits, how big and thin it is,
   what the same vessel type is built from.
3. The model's own material data: names, metallic, roughness, alpha. An artist set
   these for the look in daylight; they break a tie, they do not decide one. A
   metallic map often marks painted steel, and an alpha texture is a cutout as often
   as it is glass; opaque glass exists too, as a dark, flat band of windows on a house.

Do not add what the model does not show: no funnel on a vessel without a stack, no
glass where windows are only painted into a wall's texture.

## Describing it as simply as is right

Pick the coarsest description that is still true, and refine only where a sheet shows
it wrong:

1. A zone per material, the one covering most of its area. Most assets need nothing
   else.
2. A rule, where one material holds two zones and a property of the model separates
   them, as a metallic atlas separates rails from hull. Write it from the numbers in
   the JSON, then check it on the sheet: a rule that also catches what it should not is
   worse than none.
3. Single islands, last, for what neither catches. Listed islands are tied to the
   import's numbering, which the tests guard.

## Proving it

Render the zones sheet and look at every view: each surface in the zone it should be,
nothing in a zone it should not. Run `uv run pytest tests/test_zones.py`, which checks
the table against the import. Hand the sheet to a person, and say which parts you were
unsure of and why.

---
name: frames
description: Coordinate frames, poses and angle conventions, the robotics way: one internal convention, transforms named by their frames and composed by matching names, other conventions reached by a change of basis where a file is written. Use when code builds, composes or reads back a rotation, a pose, a bearing or an Euler angle, or writes one to a format with its own convention (OpenCV, Blender, an Oceanus recording, NMEA). Where code lives is `architecture`'s.
---

# Frames

Every angle and vector belongs to a frame. Name the frame, compose by frames, and convert conventions once, at the edge.

## The frames

Every body is RFU (+X starboard or right, +Y bow or forward, +Z up); `world` is ENU. A body at heading 0 has the world's axes.

| Frame | Standard | Origin | Axes |
|---|---|---|---|
| world | REP 105 `map`; Fossen {n}, with ENU axes | the tangent plane at mean sea level under the ownship's origin at the start | +X east, +Y north at the origin (the ownship's bow at the start), +Z up |
| level | REP 120 `base_footprint` | a hull's track point on the curved sea | world turned by the hull's heading |
| ownship, target | REP 105 `base_link` (FLU there), Fossen {b} | the asset's centre (its bounding box's; a turbine's own origin), on the design waterline | RFU, fixed to the hull |
| base | `ptu_base_link` | the rig's mount point in the ownship | ownship turned by the rig's yaw |
| head | `ptu_mount_link` | the base's point | base turned by pan, then tilted by the rig's pitch plus tilt; the IMU sits here |
| camera | OpenCV, REP 103 `*_optical` | the projection centre | right, down, along the optical axis |
| pixel | OpenCV image | the centre of the top-left pixel | u right, v down |

- **ownship and target are one kind of frame,** a hull.
- **level to hull** carries the sea's tilt, the scripted trim and list, and heave. It pivots at the floating waterline, not the hull's origin.
- **`base` is where the PTU's pan reads zero.** A scenario's tilt counts from the rig's pitch; the PTU's encoder, and so a recording's tilt, from level. A mount bolted off the bow and a moved home are the same base, and any value renders right.
- **A fixed rig** is a head with pan and tilt at 0.

Any other frame is local to the module that owns it and is named there.

## Angles

| Quantity | Convention |
|---|---|
| bearing | clockwise from world +Y |
| azimuth | the same angle in another frame (ownship, base or head). A relative bearing is an azimuth from the bow, in the level plane |
| attitude | `R = Rz(-yaw) @ Rx(pitch) @ Ry(roll)`: yaw clockwise, pitch bow up, roll starboard down. This is Fossen's ZYX with SNAME's signs |
| pan, tilt | pan clockwise, tilt up |
| wrap | (-180, 180] |

Internally an elevation is up from the horizon. The `angles` module holds the conversions and decompositions; use it rather than writing one. Other conventions are formats, never internal state:

| Format | Axes | Where it applies |
|---|---|---|
| OpenCV | right, down, forward | calibration files: an extrinsic is `T_f_camera`, the inverse of OpenCV's `[R\|t]` |
| COCO | pixel edges: OpenCV + 0.5 | labels |
| Blender camera | right, up, backward | a matrix handed to Blender |
| FLU (ROS REP 103) | forward, left, up: yaw to port, pitch nose down | an Oceanus recording's pan, tilt and camera rpy |
| Oceanus IMU | roll starboard down, pitch nose down, yaw clockwise: no change of basis gives it | the recording's IMU columns |
| NED (NMEA 2000) | forward, right, down: yaw clockwise, pitch nose up | NMEA attitude and heading |

## Rules

1. **Name a transform by its frames.**
   - `T_a_b` is frame b's pose in frame a; it maps b's coordinates into a. `T_world_camera = T_world_ownship @ T_ownship_head @ T_head_camera`.
   - Frame names are whole words, one per frame. A product whose inner names don't match is the bug.
   - An inverse gets its own name, `T_camera_world`.
   - `R_a_b` is the rotation alone and `p_a_b` the translation: `p_a = R_a_b @ p_b + p_a_b`.
   - Name frame kinds, never instances: `T_head_camera[name]`.
   - `ruff.toml` lets these capitalised names through pep8-naming.
2. **A vector or angle says its frame,** in its name or its docstring: `up_deg` of what, above which horizon.
3. **Compose matrices, never angles.** Adding a rig's yaw to a camera's yaw is AGENTS.md's trap; it holds only on a level, centred chain. Build each link as a matrix, multiply, read back once.
4. **Euler angles are a readout,** taken from a finished rotation by one decomposition, never stored and fed back in.
5. **Convert a convention by a change of basis,** with a named constant, then decompose.
   - Re-axing both frames is `R_a'_b' = B_a @ R_a_b @ B_b.T`, and `B @ R @ B.T` only when both take the same `B`.
   - Re-axing the child alone is `R_a_b @ B_b.T`, as a camera's pose does to reach OpenCV.
   - A sign flip per field hides which frame the field lives in; it is the symptom of a missing basis. Where no basis gives a convention, as with the Oceanus IMU, flip once where the file is written, by a named constant.
6. **Convert at the boundary.** The code that writes a file, or hands a matrix to Blender, converts; everything before it stays in the internal convention.
7. **Draw the chain** in a docstring or the PR when a change adds a link: which frames, which way each arrow points.

## Checks

- A round trip: build a chain, read it back, get the inputs. Test off-axis and pitched together: a centred, level chain hides most mistakes.
- A refactor of a chain dumps every shipped scenario's camera `matrix_world` before and after, and diffs any file it writes.

## Reviewing

Report a table, the finding whose fix removes a hand-written flip or angle sum first:

| # | Where | Verdict | Finding | Fix |
|---|---|---|---|---|
| 1 | `file:line` | basis | three sign flips for one format | its frame's basis matrix, then one decomposition |

- **Verdict:** **name** (a transform or angle without its frames), **compose** (angles added where matrices multiply), **basis** (per-field flips for a frame change), **boundary** (a convention converted away from where it is written), **keep**. A keep gives its reason in Fix.

End with what you did not read.

Sources are in [references/sources.md](references/sources.md).

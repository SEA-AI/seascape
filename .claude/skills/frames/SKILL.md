---
name: frames
description: Coordinate frames, poses and angle conventions in seascape, the robotics way: one internal convention, transforms named by their frames and composed by matching letters, other conventions reached by a change of basis where a file is written. Use when code builds, composes or reads back a rotation, a pose, a bearing or an Euler angle, or writes one to a format with its own convention (OpenCV, Blender, Core-Backend, NMEA). Where code lives is `architecture`'s.
---

# Frames

Every angle and vector belongs to a frame. Name the frame, compose by frames, and convert conventions once, at the edge.

## The frames

| Frame | Axes | Built by |
|---|---|---|
| world | +X east, +Y north, +Z up, at sea level | `motion.track`, `motion.tilted` |
| vessel | +X starboard, +Y bow, +Z up, at the waterline under midships | the hull's pose |
| rig | the vessel's axes turned to the rig's axis, at its mount | `Rig.pose(pan, tilt)` |
| camera | OpenCV: +X right, +Y down, +Z along the optical axis | `Camera.pose()` |

Inside seascape, a bearing is clockwise from north or the bow and an elevation is up from the horizon (`angles`). Other conventions are formats, never internal state:

| Format | Frame | Where it applies |
|---|---|---|
| OpenCV | right, down, forward | calibration extrinsics; `angles.TO_CV` |
| Blender camera | right, up, backward | `blend.BLENDER_TO_CV` |
| FLU (ROS REP 103) | forward, left, up: yaw to port, pitch nose down | Core-Backend's pan, tilt and camera rpy |
| NED (aerospace, NMEA 2000) | forward, right, down: yaw clockwise, pitch nose up | NMEA attitude and heading |

## Rules

1. **Name a transform by its frames.** `T_a_b` is frame b's pose in frame a: it takes coordinates in b to coordinates in a. Compose by matching inner letters, `T_w_c = T_w_v @ T_v_r @ T_r_c`; an inverse swaps them, `T_b_a = inv(T_a_b)`. A product whose inner letters don't match is the bug.
2. **A vector or angle says its frame,** in its name or its docstring: `up_deg` of what, above which horizon.
3. **Compose matrices, never angles.** Adding a rig's yaw to a camera's yaw is AGENTS.md's trap; it holds only on a level, centred chain. Build each link with `angles.turned` or a `.pose`, multiply, read back once.
4. **Euler angles are a readout,** taken from a finished rotation by one decomposition (`angles.attitude_deg`, `angles.bearing_deg`, `angles.elevation_deg`, `angles.pitch_to_deg`), never stored and fed back in.
5. **Convert a convention by a change of basis,** `B @ R @ B.T` with a named constant (`TO_CV`, `BLENDER_TO_CV`), then decompose. A sign flip per field hides which frame the field lives in; it is the symptom of a missing basis.
6. **Convert at the boundary.** The code that writes a file, or hands a matrix to Blender, converts; everything before it stays in seascape's convention.
7. **Draw the chain** in a docstring or the PR when a change adds a link: which frames, which way each arrow points.

## Checks

- A round trip: build with `turned` or a `.pose`, read back, get the inputs. Test off-axis and pitched together: a centred, level chain hides most mistakes.
- A refactor of a chain dumps every shipped scenario's camera `matrix_world` before and after, and diffs any file it writes.

## Reviewing

Report a table, the finding whose fix removes a hand-written flip or angle sum first:

| # | Where | Verdict | Finding | Fix |
|---|---|---|---|---|
| 1 | `file:line` | basis | three sign flips for one format | its frame's basis matrix, then one decomposition |

- **Verdict:** **name** (a transform or angle without its frames), **compose** (angles added where matrices multiply), **basis** (per-field flips for a frame change), **boundary** (a convention converted away from where it is written), **keep**. A keep gives its reason.

End with what you did not read.

Sources are in [references/sources.md](references/sources.md).

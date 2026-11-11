"""What the server tells a client about a user who has never seen a scenario file."""

# A wish and the `render` overrides that grant it. Examples, not physics.
WISHES: dict[str, list[str]] = {
    "a backlit ship": ["sky.sun_bearing_deg = 0"],
    "a rougher sea": ["sea.wind_speed_mps = 14"],
    "fog": ["sky.visibility_km = 1.5"],
    "a yacht 300 m off the bow": [
        'objects = [{ asset = "yacht", range_m = 300, bearing_deg = 0 }]'
    ],
    "a Watchkeeper 1.5 m up, as on a small boat": [
        'rigs.bow.preset = "watchkeeper"',
        "rigs.bow.height_m = 1.5",
    ],
    "a Sentry mounted 1 m high": [
        'rigs.bow.preset = "sentry"',
        "rigs.bow.height_m = 1",
    ],
    "three Watchkeepers for an all-round view": [
        'rigs.bow.preset = "watchkeeper"',
        "rigs.bow.height_m = 3",
        'rigs.port.preset = "watchkeeper"',
        "rigs.port.height_m = 3",
        "rigs.port.yaw_deg = -120",
        'rigs.stbd.preset = "watchkeeper"',
        "rigs.stbd.height_m = 3",
        "rigs.stbd.yaw_deg = 120",
    ],
}

INSTRUCTIONS = (
    "The user may never have seen a scenario or TOML: turn what they ask for into "
    "`render` overrides yourself and say in plain words what you changed. Read "
    "`catalog`, then `scenario` and `schema` for what you will change. Each `render` "
    "takes every override again, and a list such as `objects` replaces the whole "
    "list. Overrides for common wishes, over the default: "
    + "; ".join(
        f"{wish}: {', '.join(f'`{line}`' for line in lines)}"
        for wish, lines in WISHES.items()
    )
    + ". Without draws, variants differ only in their waves: render `randomized`, or "
    "draw fields in `overrides`, for varied images."
)


def getting_started() -> str:
    """A first frame, then changes to try."""
    return (
        "I have not used seascape before. Render the default scenario, one frame, and "
        "show it with what is in it, in plain words. Then offer changes to try, in "
        f"everyday words, such as {', '.join(WISHES)}. When I pick one, keep my "
        "earlier picks, render again and say what changed. Never ask me for TOML."
    )

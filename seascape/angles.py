"""Angles in degrees."""


def wrapped_deg(angle_deg: float) -> float:
    """The same direction in [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0

"""Settings every test runs under."""

import bpy
import pytest


@pytest.fixture(autouse=True)
def _generic_kernels() -> None:
    """Cycles specializes a kernel to each scene's shaders and keeps it on disk. A test
    renders its scene once, and every shader edit makes each test scene new, so the
    suite would only fill that cache. Per test: factory settings put it back."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.kernel_optimization_level = "OFF"

"""Render scenarios for MCP clients, one job at a time on this machine's GPU."""

from seascape.server.settings import Settings
from seascape.server.transport import app, mcp, serve
from seascape.server.work import State

__all__ = ["Settings", "State", "app", "mcp", "serve"]

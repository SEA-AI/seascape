"""The MCP server and the HTTP app that serves it."""

import asyncio
import subprocess
import sys
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles

from seascape.server.jobs import State, sweep
from seascape.server.tools import (
    cancel,
    catalog,
    jobs,
    render,
    render_result,
    scenario,
    schema,
)

# A judgement: longer than a driver takes to answer, short of a hung one.
PROBE_S = 60.0


def mcp(state: State) -> MCPServer:
    """The MCP server, its tools sharing `state`."""
    state.renders.mkdir(parents=True, exist_ok=True)

    # The lifespan runs per connection; jobs outlive one.
    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncGenerator[State]:
        yield state

    served = MCPServer(
        "seascape",
        instructions="Synthetic maritime camera frames with exact ground truth. Read "
        "`catalog` once, then `scenario` and `schema` for what you will change. Start "
        "a job with `render`, then call `render_result` until it is done. Without "
        "draws, variants differ only in their waves: for varied images, render "
        "`randomized`, or draw fields in `overrides`.",
        lifespan=lifespan,
    )
    # Only `render` and `cancel` change what the server holds.
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    served.add_tool(catalog, annotations=read_only)
    served.add_tool(jobs, annotations=read_only)
    served.add_tool(scenario, annotations=read_only)
    served.add_tool(schema, annotations=read_only)
    served.add_tool(
        render,
        annotations=ToolAnnotations(destructive_hint=False, open_world_hint=False),
    )
    served.add_tool(
        cancel,
        annotations=ToolAnnotations(destructive_hint=True, open_world_hint=False),
    )
    # Unstructured: the previews are images, which structured content cannot carry.
    served.add_tool(render_result, annotations=read_only, structured_output=False)
    return served


def serve(state: State, host: str, port: int) -> None:
    """Serve `app` on `host`:`port`."""
    # In a process of its own, so the server never holds Blender.
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from seascape.blender import compositor; "
                "print(compositor.enable_gpu())",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=PROBE_S,
        )
    except subprocess.TimeoutExpired:
        print(f"The GPU probe gave no answer in {PROBE_S:g} s", flush=True)
    except subprocess.CalledProcessError as error:
        # Serving anyway: each job reports what the render itself finds.
        print(f"The GPU probe failed: {error.stderr.strip()[-500:]}", flush=True)
    else:
        words = probe.stdout.split()
        backend = words[-1] if words else "None"
        print(
            f"Cycles renders on {'the CPU' if backend == 'None' else backend}",
            flush=True,
        )
    # The access log would write the secret path.
    uvicorn.run(
        app(state, host),
        host=host,
        port=port,
        access_log=state.settings.mcp_secret is None,
    )


def app(state: State, host: str) -> Starlette:
    """The MCP server and the HTTP app that serves it."""
    secret = state.settings.mcp_secret
    path = f"/mcp/{secret.get_secret_value()}" if secret else "/mcp"
    # A web page can point a browser on the network at this port by DNS rebinding,
    # so MCP refuses any Host or Origin not listed.
    hosts = list(state.settings.allowed_hosts)
    if tunnel := state.settings.public_url:
        # `port` is the scheme's default when the URL names none; a Host may carry it.
        hosts += [tunnel.host or "", f"{tunnel.host}:{tunnel.port}"]
    security = TransportSecuritySettings(
        allowed_hosts=hosts,
        allowed_origins=[f"{s}://{h}" for h in hosts for s in ("http", "https")],
    )
    served = mcp(state).streamable_http_app(
        host=host, streamable_http_path=path, transport_security=security
    )
    served.mount("/renders", StaticFiles(directory=state.renders))
    # Wrap, don't replace: the app's lifespan runs the MCP session manager.
    mcp_lifespan = served.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[Mapping[str, Any]]:
        sweeping = asyncio.create_task(sweep(state))
        try:
            async with mcp_lifespan(app) as held:
                yield held or {}
        finally:
            sweeping.cancel()
            with suppress(asyncio.CancelledError):
                await sweeping

    served.router.lifespan_context = lifespan
    return served

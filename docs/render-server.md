# Render server

`seascape serve` renders for MCP clients, so anyone on the network can ask Claude for frames without installing seascape. Jobs run one at a time on the server's GPU.

## Connect

Claude Code:

```bash
claude mcp add --transport http seascape http://<host>:8765/mcp
```

Claude Desktop, through [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "seascape": {
      "command": "npx",
      "args": ["mcp-remote", "http://<host>:8765/mcp", "--allow-http"]
    }
  }
}
```

Then ask for what you want: *"render the baseline at sunset with a yacht 300 m off the bow"*, or *"8 variants of the randomized scenario"*.

A job's files are served at `http://<host>:8765/renders/<job>/` and zipped at `http://<host>:8765/renders/<job>.zip`. `jobs` lists the jobs still kept, with their download links, and the disk left. `cancel` stops a queued or running job and deletes what it wrote.

A job past the limits in [`seascape/server.py`](../seascape/server.py) is refused, as is one whose images and archive, at their largest, would leave less than `SEASCAPE_MIN_FREE_GB` beside the jobs queued.

> [!WARNING]
> No authentication: anyone who can reach the port can queue renders and download what they write. Keep it on a trusted network.

MCP answers only the hosts in `SEASCAPE_ALLOWED_HOSTS`, which stops a web page reaching it through a browser on the network. Unset, that is this machine alone, so a shared server lists the names clients use, such as `<host>:*,<ip>:*`.

## Run it

On a machine with an NVIDIA GPU, Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/):

```bash
mkdir seascape && cd seascape
curl -fsSLO https://raw.githubusercontent.com/SEA-AI/seascape/main/compose.yaml
docker compose up -d
docker compose logs   # names the GPU backend
```

To update, `docker compose pull && docker compose up -d`. From a checkout, `SEASCAPE_COMMIT=$(git describe --always --dirty --abbrev=40 --exclude='*') docker compose up -d --build` builds it instead, and its labels name that commit, marked `-dirty` if the tree has changes.

Compose reads these from the shell or from a `.env` beside `compose.yaml`:

| variable | sets |
|---|---|
| `SEASCAPE_RENDERS` | the renders' folder, `renders/` beside `compose.yaml` unless set |
| `SEASCAPE_PORT` | the port, in place of 8765 in the URLs above |
| `SEASCAPE_MAX_DURATION_S`, `SEASCAPE_MAX_PIXELS`, `SEASCAPE_MAX_IMAGES`, `SEASCAPE_MIN_FREE_GB` | a job's limits, per camera for pixels |
| `SEASCAPE_MAX_RUN_S` | how long a job runs before it fails |
| `SEASCAPE_MAX_QUEUED` | how many jobs wait or run at once |

```bash
echo "SEASCAPE_MAX_DURATION_S=120" >> .env
docker compose up -d
```

Without Docker, `uv run seascape serve --host 0.0.0.0` serves the checkout's `scenarios/`, or `--scenarios DIR` (`SEASCAPE_SCENARIOS`).

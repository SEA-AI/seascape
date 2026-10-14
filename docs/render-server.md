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

A job's files are served at `http://<host>:8765/renders/<job>/`, with everything in `<job>.zip`.

> [!WARNING]
> No authentication: anyone who can reach the port can queue renders and download what they write. Keep it on a trusted network.

## Run it

On a machine with an NVIDIA GPU, Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/):

```bash
git clone https://github.com/SEA-AI/seascape.git
cd seascape
docker compose up -d --build
docker compose logs   # names the GPU backend
```

To update, `git pull` and run `docker compose up -d --build` again.

Without Docker, `uv run seascape serve --host 0.0.0.0` serves from a checkout.

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

A job's files are served at `http://<host>:8765/renders/<job>/`, with everything in `<job>.zip`. `jobs` lists the jobs since the server started, and the disk left.

A job past the limits at the top of [`seascape/server.py`](../seascape/server.py) is refused.

> [!WARNING]
> No authentication: anyone who can reach the port can queue renders and download what they write. Keep it on a trusted network.

## Run it

On a machine with an NVIDIA GPU, Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/):

```bash
mkdir seascape && cd seascape
curl -fsSLO https://raw.githubusercontent.com/SEA-AI/seascape/main/compose.yaml
docker compose up -d
docker compose logs   # names the GPU backend
```

To update, `docker compose pull && docker compose up -d`. From a checkout, `docker compose up -d --build` builds it instead.

Compose reads these from the shell or from a `.env` beside `compose.yaml`:

| variable | sets |
|---|---|
| `SEASCAPE_RENDERS` | the renders' folder, `renders/` beside `compose.yaml` unless set |
| `SEASCAPE_PORT` | the port |
| `SEASCAPE_MAX_DURATION_S`, `SEASCAPE_MAX_PIXELS`, `SEASCAPE_MAX_IMAGES`, `SEASCAPE_MIN_FREE_GB` | a job's limits, per camera for pixels |
| `SEASCAPE_MCP_SECRET` | moves MCP from `/mcp` to `/mcp/<secret>` |

```bash
echo "SEASCAPE_MAX_DURATION_S=120" >> .env
docker compose up -d
```

Without Docker, `uv run seascape serve --host 0.0.0.0` serves from a checkout.

## Reach it from claude.ai

A claude.ai connector calls from Anthropic's servers, so it needs a public HTTPS URL. [ngrok](https://ngrok.com) gives one by dialing out, with no inbound port. With an ngrok account, add its authtoken and dev domain to `.env`, and a secret:

```bash
echo "NGROK_AUTHTOKEN=<authtoken>" >> .env
echo "NGROK_URL=https://<dev domain>" >> .env
echo "SEASCAPE_MCP_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
docker compose --profile ngrok up -d
```

Then add a custom connector at `https://<dev domain>/mcp/<secret>` with **No sign-in**. On a Team or Enterprise plan an Owner adds it under **Organization settings > Connectors**, once for everyone.

> [!WARNING]
> The URL is the only credential: anyone who has it can queue renders. Share it as a password and change the secret if it leaks. Clients on the network need the same path.

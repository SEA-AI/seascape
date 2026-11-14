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

With `SEASCAPE_MCP_SECRET` set, each client sends it. Claude Code adds `--header "Authorization: Bearer <secret>"`. `mcp-remote` takes the value from an environment variable, which also works where a space in `args` is mangled:

```json
{
  "mcpServers": {
    "seascape": {
      "command": "npx",
      "args": ["mcp-remote", "http://<host>:8765/mcp", "--allow-http", "--header", "Authorization:${SEASCAPE_AUTH}"],
      "env": { "SEASCAPE_AUTH": "Bearer <secret>" }
    }
  }
}
```

Then ask for what you want: *"render a yacht 300 m off the bow"*, or *"8 variants of the randomized scenario"*.

New to it? Run the `getting_started` prompt: `/mcp__seascape__getting_started` in Claude Code, or the **+** menu in Claude Desktop.

With no scenario named, `render` draws `showcase`: a worn freighter at sunset. An override a scenario ignores comes back in the result's `warnings`.

A job's files are served under `http://<host>:8765/renders/<job>/` and zipped at `http://<host>:8765/renders/<job>.zip`. The archive carries no mark. `get_job` shows a marked sheet per variant (none for exr), and `preview` shows one camera's frame or one variant's sheet.

A job is refused when any seed it may try breaks a limit in the table below, or when its images and archive would leave less than `SEASCAPE_MIN_FREE_GB` free after the queued jobs.

> [!WARNING]
> Without `SEASCAPE_MCP_SECRET`, anyone who can reach the port can queue renders and download what they write. Keep it on a trusted network.

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
| `SEASCAPE_MAX_VARIANTS`, `SEASCAPE_MAX_SAMPLES` | the most variants a job renders, and the most Cycles samples per pixel |
| `SEASCAPE_MAX_RUN_S` | how long a job runs before it fails |
| `SEASCAPE_MAX_QUEUED` | how many jobs wait or run at once |
| `SEASCAPE_MCP_SECRET` | 32 or more printable ASCII characters; MCP then needs `Authorization: Bearer <secret>` |
| `SEASCAPE_PUBLIC_URL` | a tunnel's public URL, whose host MCP then answers; needs `SEASCAPE_MCP_SECRET` and `https://`; Compose sets it from `NGROK_URL` |

```bash
echo "SEASCAPE_MAX_DURATION_S=120" >> .env
docker compose up -d
```

Without Docker, `uv run seascape serve --host 0.0.0.0` serves the checkout's `scenarios/`, or `--scenarios DIR` (`SEASCAPE_SCENARIOS`).

## Reach it from claude.ai

A claude.ai connector calls from Anthropic's servers, so it needs a public HTTPS URL. [ngrok](https://ngrok.com) gives one by dialing out, with no inbound port. Add your ngrok authtoken and dev domain to `.env`, plus a secret:

```bash
echo "NGROK_AUTHTOKEN=<authtoken>" >> .env
echo "NGROK_URL=https://<dev domain>" >> .env
echo "SEASCAPE_MCP_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
docker compose --profile ngrok up -d
```

Then add a custom connector at `https://<dev domain>/mcp` with **No sign-in** and the request header `Authorization` set to `Bearer <secret>`. On a Team or Enterprise plan an Owner adds it under **Organization settings > Connectors**, once for everyone.

> [!WARNING]
> The secret is the only credential: anyone with it can queue renders. Rotate it if it leaks. `/renders` links need no secret, only their random job ID. ngrok's Traffic Inspector records requests: limit who opens it. The tunnel admits only Anthropic's [outbound ranges](https://platform.claude.com/docs/en/api/ip-addresses) on `/mcp`; other addresses get 403.

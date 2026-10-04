# Blender MCP

Lets an AI agent inspect and edit whatever scene you have open in Blender. Rendering from the CLI never touches it, so skip this unless you want the interactive workflow.

1. **Add the connector.** In Claude Desktop: **Customize → Connectors**, search *Blender*, click **Add**. It's first-party, so there's no config file and no `.mcpb`.
2. **Install the Blender add-on.** Open the [add-on install page](https://www.blender.org/lab/mcp-server/#add-on) next to Blender and drag the install link onto the Blender window **twice**: the first drop allows the Blender Lab extension repository and the second installs the add-on.
3. **Start it.** In Blender: **Edit → Preferences → Add-ons**, find *BlenderMCP*, enable **start MCP server**. Then **Save Preferences**, or it's gone on restart.

Check it's listening:

```bash
lsof -nP -iTCP:9876 -sTCP:LISTEN
```

> [!WARNING]
> The add-on runs generated code in your Blender session with no sandbox, and the port is unauthenticated. Changes only persist when you save in Blender.

<details>
<summary>Troubleshooting</summary>

| Symptom | Cause |
|---|---|
| Add-on gone after restarting Blender | Preferences were never saved. Run **Save Preferences**, or enable auto-save. |
| "Online access must be enabled" | **Edit → Preferences → System → Network → Allow Online Access**. |
| Nothing listening on 9876 | Blender isn't running, or the add-on is disabled. MCP needs the GUI. |
| Listening, but the wrong scene answers | Another Blender instance bound the port first. Only one can hold it. |
| Dragging the link does nothing | Drop it twice: the first drop only registers the repository. |
| A guide tells you to run `uvx blender-mcp` | That's [`ahujasid/blender-mcp`](https://github.com/ahujasid/blender-mcp), a different community server. Both work; don't mix their instructions. |

</details>

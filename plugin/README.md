# papervault plugin (installable package)

This directory is the minimal installable plugin shell for Claude Code and
Antigravity (`agy`) CLI/IDE. Both register a running papervault MCP server at
`http://127.0.0.1:8080/mcp` by default.

Install and start the server first — full steps are in `docs/INSTALL.md` of the
papervault repository (this installed copy contains only the plugin shell).

**Claude Code** — uses `.claude-plugin/plugin.json` and `.mcp.json`:

```
/plugin marketplace add <path-or-git-url-of-this-repo>
/plugin install papervault
```

Set `PAPERVAULT_MCP_URL` in the client environment before starting Claude Code
to use a different endpoint, including a remote tailnet URL.

**Antigravity (`agy`) CLI/IDE** — uses `plugin.json` and `mcp_config.json`.
From the repository root:

```bash
agy plugin validate ./plugin --json
agy plugin install ./plugin
agy plugin list
```

Start a new CLI or IDE session. In the IDE, inspect **Additional Options (...) >
MCP Servers** to confirm that `search_papers`, `get_paper`, and `query` are
available from the running server.

The native AGY config uses a fixed `serverUrl`; `PAPERVAULT_MCP_URL` applies to
the Claude Code config. For a custom endpoint, register MCP directly instead
of installing the AGY plugin:

```bash
agy mcp add papervault http://127.0.0.1:9090/mcp
agy mcp list
```

Replace the URL with your server's full `/mcp` endpoint. For IDE configuration,
merge the same `mcpServers.papervault.serverUrl` entry into
`~/.gemini/config/mcp_config.json`, preserving existing servers, and start a
new session. See the repository's install guide for the full JSON example.

Version policy: both plugin manifests follow the repo's single package version,
together with `.claude-plugin/marketplace.json`, `pyproject.toml`, and `uv.lock`
(stated in the repo README's Releases section). The version sync test guards all
five sources.

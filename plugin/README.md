# papervault plugin (installable package)

This directory is the minimal installable Claude Code plugin: the manifest and the MCP
registration pointing at a RUNNING papervault server (`http://127.0.0.1:8080/mcp`
by default). Set `PAPERVAULT_MCP_URL` in the client environment before starting
Claude Code to use a different endpoint, including a remote tailnet URL.

The plugin does NOT bundle the server. Install and start the server first — full steps in `docs/INSTALL.md` of the papervault repository (this installed copy contains only the plugin shell). Then:

```
/plugin marketplace add <path-or-git-url-of-this-repo>
/plugin install papervault
```

Version policy: the version field here follows the repo's single package version and is bumped in every change PR, together with `.claude-plugin/marketplace.json` and `pyproject.toml` (stated in the repo README's Releases section).

# papervault plugin (installable package)

This directory is the minimal installable Claude Code plugin: the manifest and the MCP
registration pointing at a RUNNING papervault server (`http://127.0.0.1:8080/mcp`).

The plugin does NOT bundle the server. Install and start the server first — full steps in
[`../docs/INSTALL.md`](../docs/INSTALL.md). Then:

```
/plugin marketplace add <path-or-git-url-of-this-repo>
/plugin install papervault
```

Version policy: `plugin.json` version tracks the repo release tags (bumped in the release
checklist with every tag).

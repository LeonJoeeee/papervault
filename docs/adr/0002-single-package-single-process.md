# 0002 — Single package, single MCP server process

Status: Accepted (2026-07-17)

## Context
In the source monorepo the library and knowledge layers were two Python packages and two MCP server processes, coupled only by an on-disk vault contract guarded by an index-schema version gate. That split bought independent restarts (valuable in the lab's dev loop) at the price of cross-package schema lockstep and a second service for every operator to run. MinerU vLLM, Neo4j, and Postgres are separate processes regardless — this decision covers only the two MCP servers.

## Decision
Merge into one Python package (`papervault`) and one MCP server process exposing all three tools. Subsystem boundaries survive as package modules (`library/`, `knowledge/`, `mcp/`). Rejected: the two-package mirror — its main benefit (independent restarts) serves the original lab's development habits more than any operator, while its costs (schema lockstep across repos, extra moving part) land on every installation.

## Consequences
The vault schema coupling becomes internal — reader and writer are versioned together, the version-gate lockstep problem dissolves. One restart blast radius: the lab's habit of restarting the knowledge side alone goes away. Operators run one service, one port. GPU contention is unchanged (same card either way).

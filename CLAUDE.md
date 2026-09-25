# papervault — repo notes for agents

## Commands
- install (lane venv): `uv sync --extra dev`
- test: `.venv/bin/pytest -m 'not integration' -q`
- lint: `.venv/bin/ruff check src tests`

## Gotchas
- The live `papervault.service` (user unit) runs `.venv/bin/papervault-mcp` from the primary
  checkout on `main`. Never edit that checkout, its `.venv`, or its data from a lane, and never
  restart, stop, or reconfigure the service.
- A worktree needs its own venv (`uv sync --extra dev` inside it): the primary `.venv` is an
  editable install of the live checkout, so running it from a worktree tests the wrong tree.
- Version: every change PR bumps the one repo version in all declared fields — see README.md
  "Releases".

## New worktree: copy these untracked files
- none — everything load-bearing is tracked

## Record language
- English

Architecture: see docs/architecture.md — never duplicated here. Decisions: docs/adr/. Tasks: GitHub issues.

"""Generated eval artifacts live outside the source tree; bundled gold stays in eval/."""
from __future__ import annotations

import os
from pathlib import Path


def eval_path(*parts: str) -> Path:
    """Resolve an artifact under PAPERVAULT_EVAL_DIR without creating it.

    Results are <root>/<tag>.jsonl; prompts/seeds are <root>/judge/<tag>/.
    Resolve symlinks before checking both the root and the requested artifact.
    Reject source directories in any checkout, including paths reached via aliases.
    """
    root = Path(os.environ.get("PAPERVAULT_EVAL_DIR") or "/data/paper-vault/eval-results")
    root = root.expanduser().resolve()
    path = root.joinpath(*parts).resolve()
    if "src" in root.parts or "src" in path.parts:
        raise ValueError("PAPERVAULT_EVAL_DIR artifacts must be outside src/")
    if not path.is_relative_to(root):
        raise ValueError("eval artifact path must stay inside PAPERVAULT_EVAL_DIR")
    return path

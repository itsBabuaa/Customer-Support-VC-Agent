"""Entry point — delegates to app.main."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.main import entrypoint  # noqa: E402

if __name__ == "__main__":
    from livekit.agents import cli, WorkerOptions
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

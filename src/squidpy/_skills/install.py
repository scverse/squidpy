"""Install the bundled skill for a chosen agent without downloading additional files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory

_AGENTS = ("agents", "claude", "claude-science", "codex")


def _source_dir() -> Path:
    source = Path(str(resources.files("squidpy._skills").joinpath("data")))
    if not (source / "SKILL.md").is_file():
        raise RuntimeError(f"The bundled Squidpy skill is missing from {source}.")
    return source


def _destination(agent: str | None, destination: Path | None) -> Path:
    if destination is not None:
        return destination.expanduser()
    if agent in {"agents", "codex"}:
        return Path.home() / ".agents" / "skills" / "squidpy"
    if agent == "claude":
        config = os.environ.get("CLAUDE_CONFIG_DIR")
        root = Path(config).expanduser() if config else Path.home() / ".claude"
        return root / "skills" / "squidpy"
    if agent == "claude-science":
        root = Path.home() / ".claude-science"
        settings = root / "active-org.json"
        try:
            active = json.loads(settings.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read the active organization from {settings}; use --dest instead.") from error
        organization = active.get("org_uuid") if isinstance(active, dict) else None
        if (
            not isinstance(organization, str)
            or not organization
            or organization in {".", ".."}
            or "/" in organization
            or "\\" in organization
        ):
            raise ValueError(f"Invalid org_uuid in {settings}; use --dest instead.")
        return root / "orgs" / organization / "skills" / "squidpy"
    raise ValueError("Choose --agent or --dest; there is no default installation location.")


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    """Install or inspect the agent skill shipped with Squidpy.

    Parameters
    ----------
    argv
        Command-line arguments. When omitted, read the process arguments.

    Returns
    -------
    int
        Zero on success, or one when installation or verification fails.
        Invalid command-line syntax is handled by :mod:`argparse`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=_AGENTS, help="Agent whose user skills directory receives the skill.")
    parser.add_argument("--dest", type=Path, help="Exact skill directory to install into; overrides --agent.")
    parser.add_argument("--force", action="store_true", help="Replace a differing destination, including stale files.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Check for an identical installed skill without writing.")
    action.add_argument(
        "--print-path", action="store_true", help="Print the bundled skill directory without installing."
    )
    args = parser.parse_args(argv)

    try:
        source = _source_dir()
        if args.print_path:
            print(source)
            return 0
        destination = _destination(args.agent, args.dest)
        matches = destination.is_dir() and _files(destination) == _files(source)
        if args.check:
            if matches:
                print(f"The Squidpy skill at {destination} matches the bundled copy.")
                return 0
            print(f"The Squidpy skill at {destination} is missing or differs from the bundled copy.", file=sys.stderr)
            return 1
        if matches:
            print(f"The Squidpy skill is already up to date at {destination}.")
            return 0
        if destination.is_symlink():
            raise ValueError(f"Refusing to replace the symlink at {destination}; choose another --dest.")
        source_path, destination_path = source.resolve(), destination.resolve()
        if source_path.is_relative_to(destination_path) or destination_path.is_relative_to(source_path):
            raise ValueError("The installation destination must not overlap the bundled skill; choose another --dest.")
        if destination.exists() and not args.force:
            raise ValueError(f"{destination} exists and differs; pass --force to replace it.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Finish copying before removing an existing installation.
        with TemporaryDirectory(prefix=".squidpy-skill-", dir=destination.parent) as temporary:
            prepared = Path(temporary) / "squidpy"
            shutil.copytree(source, prepared)
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            prepared.rename(destination)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Cannot install the Squidpy skill: {error}", file=sys.stderr)
        return 1
    print(f"Installed the Squidpy skill at {destination}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

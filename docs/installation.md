# Installation

Squidpy requires Python version >= 3.11 to run.

## PyPI

Install Squidpy by running::

    pip install squidpy

## Conda

Install Squidpy via Conda as::

    conda install -c conda-forge squidpy

## Development version

To install Squidpy from GitHub, run::

    pip install git+https://github.com/scverse/squidpy@main

## Agent skill

Squidpy includes a `SKILL.md` and supporting references in the Python package.
They provide coding agents with guidance for spatial analysis and image workflows.
The current instructions are validated against **Squidpy 1.8.4**; they do not
establish compatibility with a different library version.

After installing a Squidpy build that includes the skill, choose an agent:

```console
squidpy-install-skills --agent codex
squidpy-install-skills --agent claude
```

Each command copies the same bundle to the selected agent's user skills directory.
There is no default agent, and installing Squidpy does not modify agent configuration.

| Agent               | Installation directory                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------- |
| `codex` or `agents` | `~/.agents/skills/squidpy/`                                                                   |
| `claude`            | `~/.claude/skills/squidpy/`, respecting `CLAUDE_CONFIG_DIR`                                   |
| `claude-science`    | The active organization's skills directory, resolved from `~/.claude-science/active-org.json` |

Use `--dest` for an exact project or custom skill directory; it overrides `--agent`:

```console
squidpy-install-skills --dest .claude/skills/squidpy
```

An identical installation is left untouched. A differing installation requires
`--force`, which replaces the destination directory and removes stale files:

```console
squidpy-install-skills --agent codex --check
squidpy-install-skills --agent codex --force
squidpy-install-skills --print-path
```

`--check` reports a missing or differing installation with exit status 1 and makes
no changes. `--print-path` reports the package's bundled skill directory and also
makes no changes. Refresh the installed copy after upgrading Squidpy.

The same `SKILL.md` and references are installed for each agent; discovery and
invocation depend on that agent. The bundle lives in
`src/squidpy/_skills/data/` in the source checkout.

## Interactive visualization

For interactive visualization with napari, please use [napari-spatialdata](https://github.com/scverse/napari-spatialdata).
The original napari plugin from Squidpy has been deprecated and replaced by napari-spatialdata,
which offers improved functionality and support for the SpatialData ecosystem.

Install napari-spatialdata with::

    pip install napari-spatialdata

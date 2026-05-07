"""Pluggable backend dispatch system for squidpy."""

from __future__ import annotations

from scverse_backends import BackendDispatcher

_dispatcher = BackendDispatcher(
    entrypoint_group="squidpy.backends",
    host_name="squidpy",
    trusted_backends={
        "rapids_singlecell": {
            "aliases": ["rapids-singlecell", "rsc", "cuda"],
            "package": "rapids-singlecell",
            "distributions": [
                "rapids-singlecell",
                "rapids-singlecell-cu12",
                "rapids-singlecell-cu13",
            ],
            "entrypoints": ["rapids_singlecell"],
            "module_prefixes": ["rapids_singlecell"],
        },
    },
    reserved_backends={
        "gpu": "Use a concrete backend alias such as 'cuda' or 'rsc'.",
    },
)

backend_dispatch = _dispatcher.backend_dispatch
settings = _dispatcher.settings
get_backend = _dispatcher.get_backend
available_backend_names = _dispatcher.available_backend_names
discover = _dispatcher.discover

__all__ = [
    "available_backend_names",
    "backend_dispatch",
    "discover",
    "get_backend",
    "settings",
]

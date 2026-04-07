"""Dispatch decorator with introspection-based argument routing."""

from __future__ import annotations

import functools
import inspect
import re
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from squidpy._backends._registry import get_backend
from squidpy._backends._settings import settings

F = TypeVar("F", bound=Callable[..., Any])

# Cache: (func_id, backend_canonical_name) -> (shared, cpu_only, gpu_only, host_defaults)
_sig_cache: dict[tuple[int, str], tuple[set, set, set, dict]] = {}


# All functions decorated with @dispatch, so we can update their signatures later
_dispatched_functions: list[Callable] = []

_BACKEND_DOC = "backend\n    Backend to use. Use ``'cpu'`` for the default implementation or a\n    registered backend name (e.g. ``'gpu'``). See ``squidpy.settings.backend``."


def _get_param_sets(
    func: Callable,
    adapter_method: Callable,
    func_name: str,
    backend_name: str,
) -> tuple[set, set, set, dict]:
    """Compute shared/cpu_only/gpu_only param sets. Cached per function+backend."""
    key = (id(func), backend_name)
    if key in _sig_cache:
        return _sig_cache[key]

    host_sig = inspect.signature(func)
    adapter_sig = inspect.signature(adapter_method)

    host_params = set(host_sig.parameters.keys()) - {"self", "args", "kwargs"}
    adapter_params = set(adapter_sig.parameters.keys()) - {"self", "args", "kwargs"}

    # Remove "backend" — it's the dispatch kwarg, not forwarded
    host_params.discard("backend")
    adapter_params.discard("backend")

    shared = host_params & adapter_params
    cpu_only = host_params - adapter_params
    gpu_only = adapter_params - host_params

    # Cache host defaults to detect non-default cpu_only args
    host_defaults = {}
    for name, param in host_sig.parameters.items():
        if name in cpu_only and param.default is not inspect.Parameter.empty:
            host_defaults[name] = param.default

    result = (shared, cpu_only, gpu_only, host_defaults)
    _sig_cache[key] = result
    return result


def _extract_param_docs(docstring: str | None, param_names: set[str]) -> dict[str, str]:
    """Extract numpydoc parameter entries for the given names.

    Returns a dict mapping param name to its full doc block (name line + indented body).
    """
    if not docstring or not param_names:
        return {}

    result: dict[str, str] = {}
    lines = docstring.split("\n")

    # Find "Parameters" section
    in_params = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "Parameters":
            # Next line should be "----------"
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                in_params = True
                i += 2
                continue
        if in_params and stripped and not stripped[0].isspace() and stripped.startswith(("Returns", "Raises", "See ", "Notes", "Examples", "Yields", "Warns", "References")):
            break
        if in_params:
            # Check if this line starts a parameter (non-indented or 4-space indented name)
            match = re.match(r"^(\s*)(\w+)\s*(?::.*)?$", lines[i])
            if match:
                indent = match.group(1)
                name = match.group(2)
                if name in param_names:
                    # Collect this param's doc block
                    block_lines = [lines[i]]
                    j = i + 1
                    while j < len(lines):
                        # Continuation lines are more indented than the param name
                        if lines[j].strip() == "":
                            block_lines.append(lines[j])
                            j += 1
                            continue
                        line_indent = len(lines[j]) - len(lines[j].lstrip())
                        name_indent = len(indent)
                        if line_indent > name_indent:
                            block_lines.append(lines[j])
                            j += 1
                        else:
                            break
                    # Strip trailing blank lines
                    while block_lines and block_lines[-1].strip() == "":
                        block_lines.pop()
                    result[name] = "\n".join(block_lines)
        i += 1

    return result


def _dedent_param_doc(doc: str) -> str:
    """Strip leading indentation from an extracted param doc block."""
    lines = doc.split("\n")
    if not lines:
        return doc
    # Find minimum indentation across non-empty lines
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return doc
    min_indent = min(indents)
    return "\n".join(line[min_indent:] if line.strip() else "" for line in lines)


def _inject_param_docs(docstring: str | None, extra_docs: dict[str, str]) -> str:
    """Inject extra parameter docs + backend doc before the Returns section."""
    if not docstring:
        return docstring or ""

    # Dedent extracted docs then combine with backend doc
    dedented = [_dedent_param_doc(doc) for doc in extra_docs.values()]
    extra_block = "\n".join(dedented + [_BACKEND_DOC]) if dedented else _BACKEND_DOC

    # Find the end of the Parameters section (before Returns/Raises/etc.)
    lines = docstring.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("Returns", "Raises", "See Also", "Notes", "Examples", "Yields", "Warns", "References"):
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                insert_idx = i
                break

    if insert_idx is not None:
        # Detect the param-name indentation used in the host docstring
        indent = ""
        for line in lines[:insert_idx]:
            # Match lines that look like param names (word at start of indented block)
            match = re.match(r"^(\s*)\w+\s*$", line)
            if match:
                indent = match.group(1)
                break

        # Re-indent the extra docs to match the host style
        indented_lines = []
        for doc_line in extra_block.split("\n"):
            if doc_line.strip():
                indented_lines.append(indent + doc_line)
            else:
                indented_lines.append("")

        lines = lines[:insert_idx] + indented_lines + [""] + lines[insert_idx:]

    return "\n".join(lines)


def _build_signature(func: Callable) -> None:
    """Build the wrapper's ``__signature__`` from the host function, adding ``backend``."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if "backend" not in sig.parameters:
        backend_param = inspect.Parameter("backend", inspect.Parameter.KEYWORD_ONLY, default="cpu", annotation=str)
        kwargs_idx = next((i for i, p in enumerate(params) if p.kind == inspect.Parameter.VAR_KEYWORD), None)
        if kwargs_idx is not None:
            params.insert(kwargs_idx, backend_param)
        else:
            params.append(backend_param)

    func.__signature__ = sig.replace(parameters=params)


def update_signatures() -> None:
    """Merge GPU-only params from discovered backends into dispatched function signatures.

    Called once after backend discovery so that ``help()`` / IDE tooltips
    show the full parameter list (CPU + GPU + backend) with documentation.
    """
    import sys

    from squidpy._backends._registry import _backends

    for wrapper in _dispatched_functions:
        func = wrapper.__wrapped__
        func_name = func.__name__
        host_sig = inspect.signature(func)
        host_param_names = set(host_sig.parameters.keys())
        host_param_names.add("backend")

        # Collect GPU-only params and their docs from all backends
        gpu_params: list[inspect.Parameter] = []
        gpu_param_names: set[str] = set()
        adapter_docs: dict[str, str] = {}
        for backend in _backends.values():
            try:
                method = getattr(backend, func_name, None)
            except Exception:  # noqa: BLE001
                continue
            if method is None:
                continue
            try:
                adapter_sig = inspect.signature(method)
            except (ValueError, TypeError):
                continue

            new_names: set[str] = set()
            for name, param in adapter_sig.parameters.items():
                if name in host_param_names or name in gpu_param_names or name in {"self", "args", "kwargs"}:
                    continue
                gpu_params.append(param.replace(kind=inspect.Parameter.KEYWORD_ONLY))
                gpu_param_names.add(name)
                new_names.add(name)

            # Extract docstrings for these new params from the adapter's docstring
            if new_names:
                param_docs = _extract_param_docs(method.__doc__, new_names)
                adapter_docs.update(param_docs)

        # --- Update signature ---
        params: list[inspect.Parameter] = []
        var_kw = None
        for p in host_sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                var_kw = p
            else:
                params.append(p)

        params.extend(gpu_params)
        params.append(inspect.Parameter("backend", inspect.Parameter.KEYWORD_ONLY, default="cpu", annotation=str))
        if var_kw is not None:
            params.append(var_kw)

        merged_sig = host_sig.replace(parameters=params)
        wrapper.__signature__ = merged_sig

        # --- Update docstring ---
        wrapper.__doc__ = _inject_param_docs(wrapper.__doc__, adapter_docs)

        # Outer decorators (deprecated_params, etc.) may have copied the initial
        # __signature__ and __doc__ via functools.wraps.  Update the outermost
        # public function so help() / IDE tooltips reflect the merged view.
        mod = sys.modules.get(func.__module__)
        if mod is not None:
            public_func = getattr(mod, func_name, None)
            if public_func is not None and public_func is not wrapper:
                public_func.__signature__ = merged_sig
                public_func.__doc__ = wrapper.__doc__


def dispatch(func: F) -> F:
    """Route a function call to the active backend or fall back to CPU.

    Apply this decorator to any squidpy public function that a backend may
    accelerate.  The decorator:

    * Injects a ``backend`` keyword argument (default ``"cpu"``).
    * Injects backend-specific parameters and their docstrings from
      discovered backends into the function signature and docstring.
    * At call time, resolves the effective backend
      (``backend`` kwarg > ``squidpy.settings.backend``) and forwards
      arguments via signature introspection.

    Argument routing (GPU path):

    * **shared** (present in both host and backend) — forwarded.
    * **backend-only** (e.g. ``use_sparse``, ``multi_gpu``) — forwarded.
    * **cpu-only at default value** — silently dropped.
    * **cpu-only at non-default value** — dropped with a warning.

    Argument routing (CPU path):

    * All host arguments are forwarded normally.
    * Backend-only arguments raise ``TypeError`` (Python's own check).

    If the active backend does not implement the decorated function, the
    call falls back to the CPU implementation transparently.

    Parameters
    ----------
    func
        The CPU implementation to wrap.

    Returns
    -------
    The wrapped function with backend dispatch.

    Examples
    --------
    >>> from squidpy._backends import dispatch
    >>> @dispatch
    ... def my_function(adata, n_jobs=None):
    ...     ...  # CPU implementation
    """
    func_name = func.__name__

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        local_backend = kwargs.pop("backend", None)
        effective = local_backend or settings.backend

        if effective == "cpu":
            return func(*args, **kwargs)

        backend = get_backend(effective)
        if backend is None:
            raise RuntimeError(
                f"Backend {effective!r} is not installed. "
                f"Install it or set squidpy.settings.backend = 'cpu'."
            )

        method = getattr(backend, func_name, None)
        if method is None:
            # Backend doesn't implement this function — fall back to CPU
            return func(*args, **kwargs)

        shared, cpu_only, gpu_only, host_defaults = _get_param_sets(func, method, func_name, backend.name)

        # Route kwargs
        adapter_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in shared or key in gpu_only:
                adapter_kwargs[key] = value
            elif key in cpu_only:
                # Warn if non-default
                if key not in host_defaults or value != host_defaults[key]:
                    warnings.warn(
                        f"{key!r} has no effect on backend {effective!r}.",
                        stacklevel=2,
                    )
            else:
                # Unknown kwarg — let the adapter deal with it
                adapter_kwargs[key] = value

        return method(*args, **adapter_kwargs)

    # Initial signature with just `backend` (before backends are discovered)
    _build_signature(wrapper)
    _dispatched_functions.append(wrapper)

    return wrapper  # type: ignore[return-value]

"""Dispatch decorator with introspection-based argument routing."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from squidpy._backends._registry import get_backend
from squidpy._backends._settings import settings

F = TypeVar("F", bound=Callable[..., Any])

# Cache: (func_qualname, backend_canonical_name) -> (shared, cpu_only, gpu_only, host_defaults)
_sig_cache: dict[tuple[str, str], tuple[set, set, set, dict]] = {}


# All functions decorated with @dispatch, so we can update their signatures later
_dispatched_functions: list[Callable] = []


def _get_param_sets(
    func: Callable,
    adapter_method: Callable,
    func_name: str,
    backend_name: str,
) -> tuple[set, set, set, dict]:
    """Compute shared/cpu_only/gpu_only param sets. Cached per function+backend."""
    key = (func.__qualname__, backend_name)
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


# numpydoc section headers that end a Parameters block
_NUMPYDOC_SECTIONS = frozenset(
    ("Returns", "Raises", "See Also", "Notes", "Examples", "Yields", "Warns", "References", "Attributes", "Methods")
)


def _find_section(lines: list[str], section: str) -> tuple[int, int] | None:
    """Find a numpydoc section, returning (header_line, first_content_line).

    Returns None if the section is not found.
    """
    for i, line in enumerate(lines):
        if line.strip() == section and i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
            return i, i + 2
    return None


def _detect_indent(lines: list[str], start: int, end: int) -> str:
    """Detect the parameter-name indentation used in a numpydoc Parameters block.

    Looks for the first non-empty, non-section-header line between start and end.
    """
    for line in lines[start:end]:
        stripped = line.lstrip()
        if stripped and stripped.split()[0].replace("*", "").replace(",", "").isidentifier():
            return line[: len(line) - len(stripped)]
    return "    "


def _extract_param_docs(docstring: str | None, param_names: set[str]) -> dict[str, str]:
    """Extract numpydoc parameter entries for the given names.

    Uses indentation-based parsing: a parameter entry starts with a line
    whose indentation matches the section's base indent, and continues
    with all subsequent lines that are blank or more deeply indented.

    Returns a dict mapping param name to its dedented doc block.
    On any parse ambiguity, the parameter is skipped rather than producing
    garbled output.
    """
    if not docstring or not param_names:
        return {}

    lines = docstring.split("\n")
    section = _find_section(lines, "Parameters")
    if section is None:
        return {}

    _, content_start = section

    # Find where the Parameters section ends
    content_end = len(lines)
    for i in range(content_start, len(lines)):
        stripped = lines[i].strip()
        if stripped in _NUMPYDOC_SECTIONS:
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                content_end = i
                break

    base_indent = _detect_indent(lines, content_start, content_end)
    base_indent_len = len(base_indent)

    # Parse parameter entries by indentation
    result: dict[str, str] = {}
    i = content_start
    while i < content_end:
        line = lines[i]

        # Skip blank lines between parameters
        if not line.strip():
            i += 1
            continue

        # A parameter name line: at base indentation, starts with an identifier
        line_indent_len = len(line) - len(line.lstrip())
        if line_indent_len != base_indent_len:
            i += 1
            continue

        # Extract the parameter name (first word, before optional " : type")
        first_token = line.strip().split()[0].rstrip(",")
        # Handle *args, **kwargs style names
        name = first_token.lstrip("*")

        # Collect the body (lines indented deeper than base)
        block_lines = [line]
        j = i + 1
        while j < content_end:
            body_line = lines[j]
            if not body_line.strip():
                block_lines.append(body_line)
                j += 1
                continue
            body_indent_len = len(body_line) - len(body_line.lstrip())
            if body_indent_len > base_indent_len:
                block_lines.append(body_line)
                j += 1
            else:
                break

        # Strip trailing blank lines
        while block_lines and not block_lines[-1].strip():
            block_lines.pop()

        if name in param_names:
            # Dedent the block to remove the base indentation
            dedented = []
            for bl in block_lines:
                if bl.strip():
                    dedented.append(bl[base_indent_len:] if len(bl) >= base_indent_len else bl)
                else:
                    dedented.append("")
            result[name] = "\n".join(dedented)

        i = j

    return result


def _inject_param_docs(docstring: str | None, extra_docs: dict[str, str]) -> str:
    """Inject extra parameter docs and ``backend`` doc into a numpydoc docstring.

    Inserts before the first non-Parameters section (Returns, Raises, etc.).
    If the Parameters section can't be found, the docstring is returned unchanged
    rather than producing garbled output.
    """
    if not docstring:
        return docstring or ""

    lines = docstring.split("\n")
    section = _find_section(lines, "Parameters")
    if section is None:
        return docstring

    _, content_start = section

    # Find insertion point: just before the next section header
    insert_idx = len(lines)
    for i in range(content_start, len(lines)):
        stripped = lines[i].strip()
        if stripped in _NUMPYDOC_SECTIONS:
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                insert_idx = i
                break

    indent = _detect_indent(lines, content_start, insert_idx)
    body_indent = indent + "    "

    # Build the extra parameter lines
    extra_lines: list[str] = []
    for doc_block in extra_docs.values():
        for doc_line in doc_block.split("\n"):
            if doc_line.strip():
                extra_lines.append(indent + doc_line)
            else:
                extra_lines.append("")

    # Always add the backend parameter doc
    extra_lines.append(f"{indent}backend")
    extra_lines.append(f"{body_indent}Backend to use. Use ``'cpu'`` for the default implementation or a")
    extra_lines.append(f"{body_indent}registered backend name (e.g. ``'gpu'``). See ``squidpy.settings.backend``.")

    lines = lines[:insert_idx] + extra_lines + [""] + lines[insert_idx:]
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


def _find_public_func(wrapper: Callable) -> Callable:
    """Find the outermost public function that wraps a dispatch wrapper.

    Walks from the module-level attribute through ``__wrapped__`` to verify
    it actually chains back to our wrapper.  Returns the outermost function
    (which may be ``wrapper`` itself if no outer decorator exists).
    """
    import sys

    func = wrapper.__wrapped__
    mod = sys.modules.get(func.__module__)
    if mod is None:
        return wrapper

    candidate = getattr(mod, func.__name__, None)
    if candidate is None or candidate is wrapper:
        return wrapper

    # Walk __wrapped__ chain to verify candidate actually wraps our wrapper
    obj = candidate
    while obj is not None:
        if obj is wrapper:
            return candidate
        obj = getattr(obj, "__wrapped__", None)

    return wrapper


def update_signatures() -> None:
    """Merge GPU-only params from discovered backends into dispatched function signatures.

    Called once automatically after backend discovery so that ``help()`` /
    IDE tooltips show the full parameter list (CPU + GPU + backend) with
    documentation.
    """
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
        merged_doc = _inject_param_docs(wrapper.__doc__, adapter_docs)

        # Update the dispatch wrapper
        wrapper.__signature__ = merged_sig
        wrapper.__doc__ = merged_doc

        # If an outer decorator (e.g. deprecated_params) copied __signature__
        # and __doc__ via functools.wraps, update it too.
        public_func = _find_public_func(wrapper)
        if public_func is not wrapper:
            public_func.__signature__ = merged_sig
            public_func.__doc__ = merged_doc


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
    ... def my_function(adata, n_jobs=None): ...  # CPU implementation
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
                f"Backend {effective!r} is not installed. Install it or set squidpy.settings.backend = 'cpu'."
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

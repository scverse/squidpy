from typing import Any

from squidpy._docs import d

try:
    from squidpy.pl.interactive.interactive import Interactive
except ImportError as e:
    _reason = str(e)

    @d.dedent
    class Interactive:  # type: ignore[no-redef]
        """
        Interactive viewer for spatial data.

        Parameters
        ----------
        %(interactive.parameters)s
        """

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(f"Unable to import the interactive module. Reason: {_reason}")

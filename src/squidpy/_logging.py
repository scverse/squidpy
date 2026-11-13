"""squidpy's logger, the ``scverse.squidpy`` child of the shared scverse logger.

One setting controls squidpy and every other package on the shared logger::

    from scverse_misc.logging import config

    config.verbosity = "info"  # default "warning"
"""

from __future__ import annotations

from scverse_misc.logging import get_logger

logg = get_logger("squidpy", timed=True)

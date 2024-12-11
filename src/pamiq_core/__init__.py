from __future__ import annotations

from importlib import metadata

# pamiq_core to pamiq-core
__version__ = metadata.version(__name__.replace("_", "-"))

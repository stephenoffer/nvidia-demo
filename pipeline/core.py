"""Compatibility shim for pipeline.core.

This module provides backward compatibility for imports from pipeline.core.
The actual implementation has been moved to pipeline.core.orchestrator.

DEPRECATED: This module will be removed in a future version.
Use one of the following instead:
- `from pipeline.core.orchestrator import MultimodalPipeline` (recommended)
- `from pipeline.core import MultimodalPipeline` (also works via core/__init__.py)
- `from pipeline import MultimodalPipeline` (preferred public API)
"""

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For type checkers, import directly
    from pipeline.core.orchestrator import MultimodalPipeline
else:
    # For runtime, use the proper import path
    from pipeline.core.orchestrator import MultimodalPipeline
    
    # Only warn if imported directly (not through core/__init__.py)
    import sys
    if sys.modules.get('pipeline.core') is not None:
        warnings.warn(
            "Importing from pipeline.core (file) is deprecated. "
            "Use 'from pipeline.core import MultimodalPipeline' (package) or "
            "'from pipeline import MultimodalPipeline' (public API) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

__all__ = ["MultimodalPipeline"]

"""Ray Data expressions support for PipelineDataFrame.

This module provides access to Ray Data's expression API for efficient
columnar operations. Expressions are preferred over lambda functions
for better performance and query optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.data.expressions import Expr

# Try to import expressions API
try:
    from ray.data.expressions import col, Expr
    
    # Expression API is available
    _EXPRESSIONS_AVAILABLE = True
except ImportError:
    # Expression API not available in this Ray version
    _EXPRESSIONS_AVAILABLE = False
    
    # Create dummy classes for type hints
    class Expr:
        """Dummy Expr class when expressions are not available."""
        pass
    
    def col(name: str) -> Expr:
        """Dummy col() function when expressions are not available."""
        raise ImportError(
            "Ray Data expressions API not available. "
            "Please upgrade Ray to a version that supports expressions."
        )


__all__ = ["col", "Expr", "_EXPRESSIONS_AVAILABLE"]


"""Central context object for storing global application state.

Provides a singleton context object that stores:
- Ray Data context (via RayContext)
- Application-level context variables
- Pipeline execution state
- User-defined context variables
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PipelineContext:
    """Central context object for storing global application state.
    
    This class provides a singleton pattern to store and access:
    - Ray Data context (via RayContext)
    - Application-level context variables
    - Pipeline execution state
    - User-defined context variables
    
    Example:
        ```python
        from pipeline.utils.context import PipelineContext
        
        # Get the global context
        ctx = PipelineContext.get()
        
        # Set application-level variables
        ctx.set("experiment_name", "my_experiment")
        ctx.set("model_version", "v1.0")
        
        # Access variables
        experiment = ctx.get("experiment_name")
        
        # Get Ray Data context
        ray_data_ctx = ctx.ray_data_context
        ```
    """
    
    _instance: Optional[PipelineContext] = None
    _context_vars: dict[str, Any]
    _ray_context: Optional[Any] = None
    
    def __init__(self):
        """Initialize PipelineContext.
        
        Private constructor - use get() to get the singleton instance.
        """
        self._context_vars = {}
        self._ray_context = None
        self._initialized = False
    
    @classmethod
    def get(cls) -> PipelineContext:
        """Get the global PipelineContext instance.
        
        Returns:
            The singleton PipelineContext instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the global context (useful for testing).
        
        Clears all context variables and resets initialization state.
        """
        if cls._instance is not None:
            cls._instance._context_vars.clear()
            cls._instance._ray_context = None
            cls._instance._initialized = False
    
    def initialize(self, **kwargs: Any) -> None:
        """Initialize the context with Ray Data context.
        
        Args:
            **kwargs: Arguments to pass to RayContext.initialize()
        
        Example:
            ```python
            ctx = PipelineContext.get()
            ctx.initialize(
                num_cpus=4,
                num_gpus=1,
                eager_free=True
            )
            ```
        """
        if self._initialized:
            logger.warning("PipelineContext already initialized, skipping")
            return
        
        try:
            from pipeline.utils.ray.context import RayContext
            
            # Initialize Ray context
            init_result = RayContext.initialize(**kwargs)
            self._ray_context = RayContext
            self._initialized = True
            
            logger.info("PipelineContext initialized successfully")
            logger.debug(f"Ray initialization result: {init_result}")
        except Exception as e:
            logger.error(f"Failed to initialize PipelineContext: {e}", exc_info=True)
            raise
    
    @property
    def ray_data_context(self) -> Optional[Any]:
        """Get the Ray Data context.
        
        Returns:
            Ray Data DataContext instance or None if not initialized
        """
        if self._ray_context is None:
            try:
                from pipeline.utils.ray.context import RayContext
                self._ray_context = RayContext
            except ImportError:
                logger.warning("RayContext not available")
                return None
        
        try:
            return self._ray_context.get_data_context()
        except Exception as e:
            logger.warning(f"Failed to get Ray Data context: {e}")
            return None
    
    @property
    def ray_context(self) -> Optional[Any]:
        """Get the RayContext class instance.
        
        Returns:
            RayContext class or None if not available
        """
        if self._ray_context is None:
            try:
                from pipeline.utils.ray.context import RayContext
                self._ray_context = RayContext
            except ImportError:
                logger.warning("RayContext not available")
                return None
        return self._ray_context
    
    def set(self, key: str, value: Any) -> None:
        """Set a context variable.
        
        Args:
            key: Variable name
            value: Variable value
        
        Example:
            ```python
            ctx.set("experiment_name", "my_experiment")
            ctx.set("model_version", "v1.0")
            ```
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key)}")
        self._context_vars[key] = value
        logger.debug(f"Set context variable: {key} = {value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a context variable.
        
        Args:
            key: Variable name
            default: Default value if key not found
        
        Returns:
            Variable value or default
        
        Example:
            ```python
            experiment = ctx.get("experiment_name", "default_experiment")
            ```
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key)}")
        return self._context_vars.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if a context variable exists.
        
        Args:
            key: Variable name
        
        Returns:
            True if variable exists, False otherwise
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key)}")
        return key in self._context_vars
    
    def remove(self, key: str) -> Any:
        """Remove a context variable.
        
        Args:
            key: Variable name
        
        Returns:
            Removed value or None if key not found
        
        Raises:
            KeyError: If key not found (use get() with default for safe access)
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key)}")
        if key not in self._context_vars:
            raise KeyError(f"Context variable '{key}' not found")
        value = self._context_vars.pop(key)
        logger.debug(f"Removed context variable: {key}")
        return value
    
    def clear(self) -> None:
        """Clear all context variables.
        
        Note: This does not reset Ray context initialization.
        """
        self._context_vars.clear()
        logger.debug("Cleared all context variables")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary.
        
        Returns:
            Dictionary containing all context variables
        
        Example:
            ```python
            ctx_dict = ctx.to_dict()
            print(ctx_dict)
            ```
        """
        result = {
            "context_vars": self._context_vars.copy(),
            "initialized": self._initialized,
        }
        
        # Add Ray context info if available
        if self._ray_context is not None:
            try:
                ray_data_ctx = self.ray_data_context
                if ray_data_ctx is not None:
                    result["ray_data_context"] = {
                        "eager_free": getattr(ray_data_ctx, "eager_free", None),
                        "target_max_block_size": getattr(ray_data_ctx, "target_max_block_size", None),
                    }
            except Exception:
                pass
        
        return result
    
    def __getitem__(self, key: str) -> Any:
        """Get context variable using dictionary syntax.
        
        Args:
            key: Variable name
        
        Returns:
            Variable value
        
        Raises:
            KeyError: If key not found
        """
        if key not in self._context_vars:
            raise KeyError(f"Context variable '{key}' not found")
        return self._context_vars[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set context variable using dictionary syntax.
        
        Args:
            key: Variable name
            value: Variable value
        """
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if context variable exists using 'in' operator.
        
        Args:
            key: Variable name
        
        Returns:
            True if variable exists, False otherwise
        """
        return self.has(key)
    
    def __repr__(self) -> str:
        """String representation of context.
        
        Returns:
            String representation
        """
        return (
            f"PipelineContext(initialized={self._initialized}, "
            f"variables={len(self._context_vars)}, "
            f"ray_context={'available' if self._ray_context else 'unavailable'})"
        )


def get_context() -> PipelineContext:
    """Get the global PipelineContext instance.
    
    Convenience function for accessing the context.
    
    Returns:
        The singleton PipelineContext instance
    
    Example:
        ```python
        from pipeline.utils.context import get_context
        
        ctx = get_context()
        ctx.set("experiment_name", "my_experiment")
        ```
    """
    return PipelineContext.get()


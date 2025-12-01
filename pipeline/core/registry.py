"""Component registry for extensible pipeline components.

Provides a registry pattern for stages, datasources, and other components,
enabling dynamic discovery and registration of custom implementations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Type, TypeVar, Callable, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentRegistry:
    """Registry for pipeline components.
    
    Enables dynamic registration and discovery of stages, datasources,
    and other extensible components.
    """
    
    def __init__(self, component_type: str) -> None:
        """Initialize component registry.
        
        Args:
            component_type: Type name for this registry (e.g., 'stage', 'datasource')
        """
        self.component_type = component_type
        self._components: Dict[str, Type[T]] = {}
        self._factories: Dict[str, Callable[..., T]] = {}
    
    def register(
        self,
        name: str,
        component: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Register a component or factory function.
        
        Can be used as a decorator or called directly.
        
        Args:
            name: Unique name for the component
            component: Component class to register
            factory: Factory function to create component instances
        
        Returns:
            Decorator function if used as decorator, otherwise the component
        
        Examples:
            ```python
            # As decorator
            @registry.register('my_stage')
            class MyStage(ProcessorBase):
                pass
            
            # Direct registration
            registry.register('my_stage', MyStage)
            
            # With factory
            registry.register('my_stage', factory=lambda: MyStage(config=...))
            ```
        """
        def decorator(cls: Type[T]) -> Type[T]:
            """Register component class."""
            self._components[name] = cls
            logger.debug(f"Registered {self.component_type} '{name}': {cls.__name__}")
            return cls
        
        if component is not None:
            self._components[name] = component
            logger.debug(f"Registered {self.component_type} '{name}': {component.__name__}")
            return component
        
        if factory is not None:
            self._factories[name] = factory
            logger.debug(f"Registered {self.component_type} factory '{name}'")
            return decorator
        
        return decorator
    
    def get(self, name: str) -> Optional[Type[T]]:
        """Get registered component class.
        
        Args:
            name: Component name
        
        Returns:
            Component class or None if not found
        """
        return self._components.get(name)
    
    def create(self, name: str, *args: Any, **kwargs: Any) -> Optional[T]:
        """Create component instance.
        
        Args:
            name: Component name
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor
        
        Returns:
            Component instance or None if not found
        """
        # Try factory first
        if name in self._factories:
            return self._factories[name](*args, **kwargs)
        
        # Fall back to class instantiation
        component_class = self._components.get(name)
        if component_class is None:
            return None
        
        return component_class(*args, **kwargs)
    
    def list(self) -> list[str]:
        """List all registered component names.
        
        Returns:
            List of registered component names
        """
        return sorted(set(self._components.keys()) | set(self._factories.keys()))
    
    def unregister(self, name: str) -> bool:
        """Unregister a component.
        
        Args:
            name: Component name to unregister
        
        Returns:
            True if component was unregistered, False if not found
        """
        removed = False
        if name in self._components:
            del self._components[name]
            removed = True
        if name in self._factories:
            del self._factories[name]
            removed = True
        
        if removed:
            logger.debug(f"Unregistered {self.component_type} '{name}'")
        
        return removed
    
    def clear(self) -> None:
        """Clear all registered components."""
        self._components.clear()
        self._factories.clear()
        logger.debug(f"Cleared all {self.component_type} registrations")


# Global registries for different component types
stage_registry = ComponentRegistry("stage")
datasource_registry = ComponentRegistry("datasource")
loader_registry = ComponentRegistry("loader")
validator_registry = ComponentRegistry("validator")

__all__ = [
    "ComponentRegistry",
    "stage_registry",
    "datasource_registry",
    "loader_registry",
    "validator_registry",
]


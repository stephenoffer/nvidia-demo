"""Version information for the pipeline package.

This module provides version information following PEP 440.
"""

__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Version metadata
__version_major__ = __version_info__[0]
__version_minor__ = __version_info__[1]
__version_patch__ = __version_info__[2] if len(__version_info__) > 2 else 0


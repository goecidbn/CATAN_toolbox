"""
CATAN: tools for matching and curating neurons across imaging sessions.
"""


from catan._version import __version__

__all__ = [
    "__version__",
]

from .core import SessionData, Remapping
from .tracking import Tracking, TrackingAnalysis, match_model
from .gui import main
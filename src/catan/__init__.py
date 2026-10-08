"""
CATAN: tools for matching and curating neurons across imaging sessions.
"""

# from catan._version import __version__
# from .core import SessionData, Remapping
# from .tracking import Tracking, TrackingAnalysis, match_model

# __all__ = [
#     "__version__",
#     "SessionData",
#     "Remapping",
#     "Tracking",
#     "TrackingAnalysis",
#     "match_model",
# ]

from catan._lazy import install_exports as _install_exports

_install_exports(
    globals(),
    __name__,
    {
        "__version__": ("._version", "__version__"),
        "SessionData": (".core.structures.session", "SessionData"),
        "Remapping": (".core.structures.remap", "Remapping"),
        "Tracking": (".tracking.neuron_tracking", "Tracking"),
        "TrackingAnalysis": (
            ".tracking.neuron_tracking_analysis",
            "TrackingAnalysis",
        ),
        "match_model": (
            ".tracking.analytics.fit_model_theoretical",
            "match_model",
        ),
    },
)

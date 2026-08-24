"""Lazy re-exports.

Imports are deferred so the package can be used from an environment that has
only one of the two modelling backends installed (PyMC-Marketing for
``mmm.model``, Meridian/TensorFlow for ``mmm.meridian_model``).
"""

import importlib

_EXPORTS = {
    "load_timeseries": "mmm.preprocessing",
    "prepare_features": "mmm.preprocessing",
    "geometric_adstock": "mmm.transformations",
    "hill_saturation": "mmm.transformations",
    "build_mmm": "mmm.model",
    "prepare_model_data": "mmm.model",
    "build_meridian_frame": "mmm.meridian_model",
    "load_input_data": "mmm.meridian_model",
    "build_model": "mmm.meridian_model",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

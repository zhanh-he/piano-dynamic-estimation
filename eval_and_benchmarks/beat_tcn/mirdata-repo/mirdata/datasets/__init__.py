"""Datasets package exports.

Expose selected datasets as attributes so that
`getattr(mirdata.datasets, name)` works for dynamic loading.
"""

# Commonly used built-ins; safe to expose
from . import gtzan_genre  # noqa: F401
from . import guitarset    # noqa: F401

# Our local Mazurka H5 dataset
from . import mazurka_h5   # noqa: F401

__all__ = [
    'gtzan_genre',
    'guitarset',
    'mazurka_h5',
]

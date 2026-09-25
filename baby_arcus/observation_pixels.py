"""Request-scoped image decoding reuse; no neural activations or cross-job cache."""
from contextlib import contextmanager
from contextvars import ContextVar
from io import BytesIO
import numpy as np
from PIL import Image

_CACHE=ContextVar('arcus_decoded_observation',default=None)


@contextmanager
def reuse_pixels():
    token=_CACHE.set({})
    try:yield
    finally:_CACHE.reset(token)


def decode(raw):
    cache=_CACHE.get()
    if cache is not None and raw in cache:return cache[raw]
    with Image.open(BytesIO(raw)) as image:
        if image.width>800 or image.height>560:raise ValueError('Oversized sensory frame')
        value=np.asarray(image.convert('RGB').resize((96,96))).copy()
    # Curiosity normally uses one frame. Bound even unexpected callers to one.
    if cache is not None:
        cache.clear();cache[raw]=value
    return value

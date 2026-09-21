"""`torch_dtype` was renamed to `dtype` in transformers 4.56.

The notebook, the Hugging Face Job container and a local checkout can easily end up on
different sides of that rename, and passing the wrong one is silently ignored rather than
raising -- which means a model loads in float32 and OOMs. Resolve it once, here.
"""

from packaging.version import Version
from transformers import __version__ as _TRANSFORMERS_VERSION

DTYPE_KW = "dtype" if Version(_TRANSFORMERS_VERSION).release >= (4, 56) else "torch_dtype"


def load(cls, name_or_path, dtype=None, **kwargs):
    if dtype is not None:
        kwargs[DTYPE_KW] = dtype
    return cls.from_pretrained(name_or_path, **kwargs)

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("pycudadecon")
except PackageNotFoundError:
    __version__ = "uninstalled"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import _libwrap as lib


try:
    from . import _libwrap as lib
except FileNotFoundError as e:
    import warnings

    t = e
    warnings.warn(f"\n{e}\n\nMost functionality will fail!\n", stacklevel=2)

    class _stub:
        def __getattr__(self, name):
            raise t

    lib = _stub()  # type: ignore


from .affine import affineGPU, deskewGPU, rotateGPU
from .deconvolution import (
    RLContext,
    decon,
    rl_cleanup,
    rl_context,
    rl_decon,
    rl_init,
)
from .otf import TemporaryOTF, make_otf

__all__ = [
    "__version__",
    "affineGPU",
    "decon",
    "deskewGPU",
    "make_otf",
    "rl_cleanup",
    "rl_decon",
    "rl_init",
    "RLContext",
    "rl_context",
    "rotateGPU",
    "TemporaryOTF",
]

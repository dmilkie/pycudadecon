import os
import tempfile
from typing import Any, Optional

import numpy as np
import tifffile as tf

from . import lib
from .util import PathOrArray, imread, is_otf

def make_otf(
    psf: str,
    outpath: Optional[str] = None,
    dzpsf: float = 0.1,
    dxpsf: float = 0.1,
    wavelength: int = 520,
    na: float = 1.25,
    nimm: float = 1.3,
    otf_bgrd: Optional[int] = None,
    krmax: int = 0,
    fixorigin: int = 10,
    cleanup_otf: bool = False,
    skewed_decon: bool = True,
    **kwargs: Any,
) -> str:
    """Generate a radially averaged OTF file from a PSF file.

    Parameters
    ----------
    psf : str
        Filepath of 3D PSF TIF
    outpath : str, optional
        Destination filepath for the output OTF
        (default: appends "_otf.tif" to filename), by default None
    dzpsf : float, optional
        Z-step size in microns, by default 0.1
    dxpsf : float, optional
        XY-Pixel size in microns, by default 0.1
    wavelength : int, optional
        Emission wavelength in nm, by default 520
    na : float, optional
        Numerical Aperture, by default 1.25
    nimm : float, optional
        Refractive indez of immersion medium, by default 1.3
    otf_bgrd : int, optional
        Background to subtract. "None" = autodetect., by default None
    krmax : int, optional
        pixels outside this limit will be zeroed (overwriting
        estimated value from NA and NIMM), by default 0
    fixorigin : int, optional
        for all kz, extrapolate using pixels kr=1 to this pixel
        to get value for kr=0, by default 10
    cleanup_otf : bool, optional
        clean-up outside OTF support, by default False

    Returns
    -------
    str
        Path to the OTF file
    """
    if outpath is None:
        outpath = psf.replace(".tif", "_otf.tif")

    if otf_bgrd and isinstance(otf_bgrd, (int, float)):
        bUserBackground = True
        background = float(otf_bgrd)
    else:
        bUserBackground = False
        background = 0.0

    interpkr = fixorigin
    lib.makeOTF(
        str.encode(psf),
        str.encode(outpath),
        int(wavelength),
        dzpsf,
        fixorigin,
        bUserBackground,
        background,
        na,
        nimm,
        dxpsf,
        krmax,
        cleanup_otf,
        skewed_decon,
        )

    return outpath


class TemporaryOTF:
    """Context manager to read OTF file or generate a temporary OTF from a PSF.

    Normalizes the input PSF to always provide the path to an OTF file,
    converting the PSF to a temporary file if necessary.

    ``self.path`` can be used within the context to get the filepath to
    the temporary OTF filepath.

    Args:
    ----
        psf (str, np.ndarray): 3D PSF numpy array, or a filepath to a 3D PSF
            or 2D complex OTF file.
        **kwargs: optional keyword arguments will be passed to the
            :func:`pycudadecon.otf.make_otf` function

    Note:
    ----
        OTF files cannot currently be provided directly as 2D complex np.ndarrays

    Raises:
    ------
        ValueError: If the PSF/OTF is an unexpected type
        NotImplementedError: if the PSF/OTF is a complex 2D numpy array

    Example:
    -------
        >>> with TemporaryOTF(psf, **kwargs) as otf:
        ...     print(otf.path)

    """

    def __init__(self, psf: PathOrArray, **kwargs: Any) -> None:
        self.psf = psf
        self.kwargs = kwargs

    def __enter__(self) -> "TemporaryOTF":
        if not is_otf(self.psf):
            self.tempotf = tempfile.NamedTemporaryFile(suffix="_otf.tif", delete=False)
            if isinstance(self.psf, np.ndarray):
                # psf is array. Save array to a file. Make otf from psf file.
                temp_psf = tempfile.NamedTemporaryFile(suffix="_psf.tif", delete=False)
                tf.imwrite(temp_psf.name, self.psf)
                make_otf(temp_psf.name, self.tempotf.name, **self.kwargs)
                try:
                    temp_psf.close()
                    os.remove(self.temp_psf.name)
                except Exception:
                    pass
            elif isinstance(self.psf, str) and os.path.isfile(self.psf):
                # psf is a file. Make otf from psf file.
                make_otf(self.psf, self.tempotf.name, **self.kwargs)
            else:
                raise ValueError(f"Did not expect PSF file as {type(self.psf)}")
            self.path = self.tempotf.name
        elif is_otf(self.psf) and os.path.isfile(str(self.psf)):
            self.path = str(self.psf)
        elif is_otf(self.psf) and isinstance(self.psf, np.ndarray):
            raise NotImplementedError("cannot yet handle OTFs as numpy arrays")
        else:
            raise ValueError("Unrecognized input for otf")
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            self.tempotf.close()
            os.remove(self.tempotf.name)
        except Exception:
            pass

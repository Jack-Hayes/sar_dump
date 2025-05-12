import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter

def _nan_boxcar(data: np.ndarray,
                window: int,
                min_count: float,
                mode: str = 'constant',
                cval: float = 0.0) -> np.ndarray:
    """
    NaN‑aware boxcar (mean) filter with minimum valid‑pixel threshold.
    Applies uniform_filter to data and mask, returning windowed means
    where count >= min_count, else NaN.
    """
    # mask and fill NaNs with zero
    mask = np.isfinite(data).astype(float)
    data_filled = np.nan_to_num(data, nan=0.0)

    # SciPy uniform_filter returns the mean over the window
    mean_data = uniform_filter(data_filled, size=window,
                               mode=mode, cval=cval)
    mean_mask = uniform_filter(mask, size=window,
                               mode=mode, cval=cval)

    # mask out windows with too few valid pixels
    result = np.where(mean_mask * window * window >= min_count,
                      mean_data, np.nan)
    return result

# TODO: return the ifg too cause why not
def calculate_coherence_fast(dsR: xr.DataArray,
                             dsS: xr.DataArray,
                             window: int = 7,
                             min_valid_ratio: float = 0.5) -> xr.DataArray:
    """
    Fast sliding‐window coherence estimator for two complex SAR xarray DataArrays.

    Parameters
    ----------
    dsR, dsS : xr.DataArray (complex64)
        Reference and secondary SAR bursts (let's stop using the terms "master" and "slave"). 
        Must share dims, coords, CRS, etc.
    window : int
        Must be odd. Defines a square boxcar of size window x window.
    min_valid_ratio : float
        Fraction of non‐NaN pixels in each window to consider output valid.

    Returns
    -------
    coherence : xr.DataArray (float32)
        Coherence in [0,1], with NaNs where insufficient data.
    """
    if window % 2 == 0:
        window += 1  # enforce odd for centering

    # make complex64 and form interferogram + intensities
    dsR_c = dsR.astype(np.complex64)
    dsS_c = dsS.astype(np.complex64)
    ifg = dsR_c * np.conj(dsS_c)  # interferogram
    I1  = np.abs(dsR_c)**2
    I2  = np.abs(dsS_c)**2

    # threshold in absolute pixel count
    min_count = min_valid_ratio * window * window

    # wrap _nan_boxcar to handle 2D arrays
    def _ufunc(data_arr):
        return _nan_boxcar(data_arr,
                           window=window,
                           min_count=min_count)

    # apply to real/imag parts and intensities
    core_dims = [['y','x']]
    mean_ifg_real = xr.apply_ufunc(
        _ufunc, ifg.real,
        input_core_dims=core_dims,
        output_core_dims=core_dims,
        vectorize=True, dask='parallelized',
        output_dtypes=[np.float32]
    )
    mean_ifg_imag = xr.apply_ufunc(
        _ufunc, ifg.imag,
        input_core_dims=core_dims,
        output_core_dims=core_dims,
        vectorize=True, dask='parallelized',
        output_dtypes=[np.float32]
    )
    mean_I1 = xr.apply_ufunc(
        _ufunc, I1,
        input_core_dims=core_dims,
        output_core_dims=core_dims,
        vectorize=True, dask='parallelized',
        output_dtypes=[np.float32]
    )
    mean_I2 = xr.apply_ufunc(
        _ufunc, I2,
        input_core_dims=core_dims,
        output_core_dims=core_dims,
        vectorize=True, dask='parallelized',
        output_dtypes=[np.float32]
    )

    # reconstruct complex mean interferogram
    mean_ifg = mean_ifg_real + 1j * mean_ifg_imag

    # Compute coherence |E{ifg}| / sqrt(E{I1} E{I2})
    num = np.abs(mean_ifg)
    den = np.sqrt(mean_I1 * mean_I2) + 1e-10           # ε for stability
    coh = num / den
    coh = coh.where(den > 1e-10).clip(0,1)             # cap and mask

    coh.name = 'coherence'
    coh.attrs.update(
        description='Optimized sliding‐window interferometric coherence',
        window_size=window,
        min_valid_ratio=min_valid_ratio
    )
    # preserve spatial metadata
    coh = coh.assign_coords(dsR.coords)
    if hasattr(dsR, 'rio'):
        coh = coh.rio.write_crs(dsR.rio.crs)          # keep CRS if present

    return coh

def calculate_coherence_numpy(ref: np.ndarray,
                              sec: np.ndarray,
                              window: int = 7,
                              min_valid_ratio: float = 0.5) -> np.ndarray:
    """
    Compute sliding‐window coherence for two 2D complex arrays,
    using 'reflect' padding so edges aren’t zero‐padded.
    """
    if window % 2 == 0:
        window += 1

    ifg   = ref * np.conj(sec)
    I1, I2 = np.abs(ref)**2, np.abs(sec)**2

    real_ifg, imag_ifg = np.real(ifg), np.imag(ifg)
    valid = np.isfinite(ifg).astype(float)

    # count of valid pixels, with reflect so edges count correctly
    count     = uniform_filter(valid,    size=window, mode='reflect')
    min_count = min_valid_ratio * window * window

    # sum of values in each window
    sum_re = uniform_filter(np.nan_to_num(real_ifg), size=window, mode='reflect')
    sum_im = uniform_filter(np.nan_to_num(imag_ifg), size=window, mode='reflect')
    sum_I1 = uniform_filter(np.nan_to_num(I1),       size=window, mode='reflect')
    sum_I2 = uniform_filter(np.nan_to_num(I2),       size=window, mode='reflect')

    # normalize sums to get means
    with np.errstate(divide='ignore', invalid='ignore'):
        m_re = sum_re * (window*window / count)
        m_im = sum_im * (window*window / count)
        m_I1 = sum_I1 * (window*window / count)
        m_I2 = sum_I2 * (window*window / count)

    mean_ifg = m_re + 1j*m_im
    num = np.abs(mean_ifg)
    den = np.sqrt(m_I1 * m_I2) + 1e-10

    coh = num / den
    coh[count < min_count] = np.nan
    return np.clip(coh, 0, 1)

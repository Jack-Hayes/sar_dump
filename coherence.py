import xarray as xr
import numpy as np

def calculate_coherence_xarray(dsR: xr.DataArray,
                               dsS: xr.DataArray,
                               window_size: int = 7,
                               min_valid_ratio: float = 0.5) -> xr.DataArray:
    """
    Compute local coherence between two complex SAR images using xarray rolling window.

    Handles NaNs by skipping them in the rolling mean calculation and
    masking output pixels where the window contains too few valid input pixels.

    Parameters
    ----------
    dsR, dsS : xr.DataArray (complex)
        Reference and secondary complex SAR images (e.g., bursts).
        Should have dimensions like ('y', 'x').
    window_size : int, optional
        Size of the square averaging window (must be odd for center=True).
        Default is 7.
    min_valid_ratio : float, optional
        Minimum fraction of valid (non-NaN) pixels required within a window
        for the output coherence pixel to be considered valid. Default is 0.5.

    Returns
    -------
    xr.DataArray
        Coherence map with values in [0, 1]. Pixels that do not meet
        the min_valid_ratio threshold will be NaN.
    """
    if window_size % 2 == 0:
        print(f"Warning: window_size {window_size} is even. Using {window_size+1} for centered window.")
        window_size += 1

    # Ensure input is complex float
    dsR = dsR.astype(np.complex64)
    dsS = dsS.astype(np.complex64)

    # 1. Interferogram product
    ifg = dsR * np.conj(dsS)

    # 2. Intensity images
    intensity_R = np.abs(dsR)**2
    intensity_S = np.abs(dsS)**2

    # Define rolling window dimensions
    window_dims = {'x': window_size, 'y': window_size}
    # center=True is an argument for .rolling()
    # skipna=True is an argument for .mean()

    # 3. Calculate mean of interferogram and intensities over the window
    # Pass skipna to the .mean() method, not .rolling()
    mean_ifg = ifg.rolling(window_dims, center=True).mean(skipna=True)
    mean_intensity_R = intensity_R.rolling(window_dims, center=True).mean(skipna=True)
    mean_intensity_S = intensity_S.rolling(window_dims, center=True).mean(skipna=True)

    # 4. Calculate coherence numerator and denominator
    coh_num = np.abs(mean_ifg)
    coh_den = np.sqrt(mean_intensity_R * mean_intensity_S)

    # 5. Compute coherence, adding epsilon to avoid division by zero
    coherence = coh_num / (coh_den + 1e-9) # Epsilon for numerical stability

    # 6. Apply mask based on the minimum number of valid pixels in the window
    valid_mask = (~dsR.isnull()) & (~dsS.isnull())
    # Sum the boolean mask to count valid pixels. skipna in .sum() won't matter
    # for a boolean mask unless the mask itself could have NaNs.
    valid_count = valid_mask.rolling(window_dims, center=True).sum()
    total_pixels_in_window = window_size * window_size
    min_valid_pixels = min_valid_ratio * total_pixels_in_window

    coherence = coherence.where(valid_count >= min_valid_pixels)

    # Add attributes
    coherence.attrs['description'] = 'Interferometric Coherence'
    coherence.attrs['window_size'] = window_size
    coherence.attrs['min_valid_ratio'] = min_valid_ratio
    coherence.name = 'coherence'

    return coherence

# https://github.com/isce-framework/isce2/blob/e1da858ddf1efb2a33fc5cc36a0c61721001b3d5/components/isceobj/Alos2Proc/runCoherence.py#L57
import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter
import warnings

# Helper function for NaN-aware uniform filtering (boxcar mean)
def _nan_uniform_filter(arr: np.ndarray, window_size: int, min_valid_fraction: float = 0.5) -> np.ndarray:
    """
    Performs uniform filtering (boxcar mean) ignoring NaNs.

    Parameters:
        arr (np.ndarray): Input array.
        window_size (int): Size of the square filter window.
        min_valid_fraction (float): Minimum fraction of valid pixels in window
                                   for the output to be non-NaN.

    Returns:
        np.ndarray: Filtered array.
    """
    if np.all(np.isnan(arr)):
        return arr # Return original if all NaN

    # Create a mask where valid pixels are 1, NaNs are 0
    valid_mask = (~np.isnan(arr)).astype(float)
    # Replace NaNs with 0 for summation
    arr_filled = np.nan_to_num(arr, nan=0.0)

    # Use uniform_filter for fast summation over the window
    sum_arr = uniform_filter(arr_filled, size=window_size, mode='constant', cval=0.0)
    sum_mask = uniform_filter(valid_mask, size=window_size, mode='constant', cval=0.0)

    # Avoid division by zero where the window has no valid pixels
    # Add a small epsilon where sum_mask is zero
    sum_mask_stable = np.where(sum_mask == 0, 1e-9, sum_mask)

    # Calculate the mean over valid pixels
    mean_arr = sum_arr / sum_mask_stable

    # Apply the minimum valid fraction threshold
    min_valid_count = min_valid_fraction * (window_size ** 2)
    result = np.where(sum_mask >= min_valid_count, mean_arr, np.nan)

    return result.astype(arr.dtype if np.issubdtype(arr.dtype, np.floating) else float) # Preserve float type

def calculate_coherence_optimized(dsR: xr.DataArray,
                                  dsS: xr.DataArray,
                                  window_size: int = 7,
                                  min_valid_ratio: float = 0.5) -> xr.DataArray:
    """
    Compute local coherence between two complex SAR images using scipy.ndimage.uniform_filter
    wrapped in xr.apply_ufunc for efficiency and xarray integration.

    Handles NaNs by skipping them during averaging and masking output pixels
    where the window contains too few valid input pixels.

    Parameters
    ----------
    dsR, dsS : xr.DataArray (complex)
        Reference and secondary complex SAR images (e.g., bursts).
        Should have dimensions like ('y', 'x'). Assumed to have matching coords.
    window_size : int, optional
        Size of the square averaging window. Default is 7.
    min_valid_ratio : float, optional
        Minimum fraction of valid (non-NaN) pixels required within a window
        for the output coherence pixel to be considered valid. Default is 0.5.

    Returns
    -------
    xr.DataArray
        Coherence map with values in [0, 1]. Pixels that do not meet
        the min_valid_ratio threshold will be NaN. Has same coordinates as inputs.
    """
    if not isinstance(dsR, xr.DataArray) or not isinstance(dsS, xr.DataArray):
        raise TypeError("Inputs dsR and dsS must be xarray DataArrays.")
    if dsR.shape != dsS.shape:
        raise ValueError("Input DataArrays dsR and dsS must have the same shape.")
    if dsR.dims != dsS.dims or len(dsR.dims) != 2:
         warnings.warn(f"Inputs dsR and dsS dims differ or are not 2D: {dsR.dims} vs {dsS.dims}. Proceeding assuming ('y', 'x').")
         # Attempt to guess y, x dimensions - adapt if needed
         y_dim = dsR.dims[0]
         x_dim = dsR.dims[1]
    else:
        y_dim = dsR.dims[0]
        x_dim = dsR.dims[1]


    # --- Define the wrapper for apply_ufunc ---
    def _filter_wrapper(arr, window_size_arg, min_valid_ratio_arg):
        # uniform_filter needs padding handled if mode='nearest' etc. is desired,
        # but for simple averaging, apply_ufunc handles chunk boundaries reasonably
        # if data is chunked. If not chunked, it's straightforward.
        # mode='constant', cval=0 is used within _nan_uniform_filter
        return _nan_uniform_filter(arr, window_size=window_size_arg, min_valid_fraction=min_valid_ratio_arg)

    # --- Prepare data ---
    # Ensure input is complex float
    dsR = dsR.astype(np.complex64)
    dsS = dsS.astype(np.complex64)

    # 1. Interferogram product
    ifg = dsR * np.conj(dsS)

    # 2. Intensity images
    intensity_R = np.abs(dsR)**2
    intensity_S = np.abs(dsS)**2

    # --- Apply filtering using xr.apply_ufunc ---
    # Filter real and imaginary parts of the interferogram separately
    mean_ifg_real = xr.apply_ufunc(
        _filter_wrapper,
        ifg.real,
        kwargs={'window_size_arg': window_size, 'min_valid_ratio_arg': min_valid_ratio},
        dask='parallelized',  # Allow dask parallelism
        output_dtypes=[ifg.real.dtype],
        keep_attrs=True # Keep coordinates etc.
    ).rename('mean_ifg_real') # Rename for clarity, apply_ufunc might strip name

    mean_ifg_imag = xr.apply_ufunc(
        _filter_wrapper,
        ifg.imag,
        kwargs={'window_size_arg': window_size, 'min_valid_ratio_arg': min_valid_ratio},
        dask='parallelized',
        output_dtypes=[ifg.imag.dtype],
         keep_attrs=True
    ).rename('mean_ifg_imag')

    # Combine filtered parts back to complex mean interferogram
    mean_ifg = mean_ifg_real + 1j * mean_ifg_imag

    # Filter intensity images
    mean_intensity_R = xr.apply_ufunc(
        _filter_wrapper,
        intensity_R,
        kwargs={'window_size_arg': window_size, 'min_valid_ratio_arg': min_valid_ratio},
        dask='parallelized',
        output_dtypes=[intensity_R.dtype],
         keep_attrs=True
    ).rename('mean_intensity_R')

    mean_intensity_S = xr.apply_ufunc(
        _filter_wrapper,
        intensity_S,
        kwargs={'window_size_arg': window_size, 'min_valid_ratio_arg': min_valid_ratio},
        dask='parallelized',
        output_dtypes=[intensity_S.dtype],
         keep_attrs=True
    ).rename('mean_intensity_S')


    # --- Calculate Coherence ---
    coh_num = np.abs(mean_ifg)
    # Ensure non-negative intensities before sqrt
    mean_intensity_R_safe = xr.where(mean_intensity_R < 0, 0, mean_intensity_R)
    mean_intensity_S_safe = xr.where(mean_intensity_S < 0, 0, mean_intensity_S)
    coh_den = np.sqrt(mean_intensity_R_safe * mean_intensity_S_safe)

    # Compute coherence, adding epsilon to avoid division by zero
    coherence = coh_num / (coh_den + 1e-9) # Epsilon for numerical stability

    # Ensure coherence is within [0, 1] and handle potential NaNs from division
    coherence = xr.where(coh_den <= 1e-9, 0, coherence) # Set coherence to 0 if denominator is near zero
    coherence = xr.where(coherence > 1.0, 1.0, coherence) # Cap at 1
    coherence = xr.where(coherence < 0.0, 0.0, coherence) # Floor at 0 (shouldn't happen ideally)
    coherence = coherence.where(~np.isnan(mean_ifg)) # Ensure NaN if mean_ifg was NaN

    # Preserve coordinates and add attributes

    #coherence = coherence.rio.write_crs(dsR.rio.crs, inplace=True) # Ensure CRS
    # MissingCRS error is common when operations inadvertently drop the special spatial_ref coordinate that rioxarray uses to track the CRS

    coherence.attrs['description'] = 'Interferometric Coherence (Optimized)'
    coherence.attrs['window_size'] = window_size
    coherence.attrs['min_valid_ratio'] = min_valid_ratio
    coherence.attrs['reference_file'] = dsR.encoding.get('source', 'Unknown') # Get source if available
    coherence.attrs['secondary_file'] = dsS.encoding.get('source', 'Unknown')
    coherence.name = 'coherence'

    return ifg, coherence

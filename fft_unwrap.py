import numpy as np
from scipy.ndimage import uniform_filter

def unwrap_phase_fft(wrapped, window=5):
    """
    Unwrap a 2D wrapped phase image using an FFT-based least-squares approach
    with an optional quality filtering pre-step.
    
    The unwrapped phase φ_unwrapped at a pixel (i, j) can be thought of as the sum
    of the wrapped phase φ_wrapped at (i, j) plus an integer multiple of 2π that
    accounts for the cumulative phase differences (Δφ) between neighboring pixels:
    
       φ_unwrapped(i, j) = φ_wrapped(i, j) + 2π ∑ₖ₌₁ⁱ ∑ₗ₌₁ʲ Δφ(k, l)
    
    This function works as follows:
      1. It replaces any NaNs with the mean of valid phase values.
      2. It computes a quality map based on the local variance of the phase gradient.
         Unreliable pixels (with high local variance) are replaced by a locally
         smoothed (circular-mean) value. This step is inspired by methods such as the
         derivative variance correlation map (Lu et al., 2007). Note that this pre-filtering
         may affect phase values near the edges.
      3. It computes finite differences (phase gradients) along x and y.
      4. It wraps these finite differences to the interval [-π, π].
      5. It expands the computed differences back to the full image size.
      6. It builds the divergence (the right-hand side of the Poisson equation) by
         taking differences of these expanded gradients.
      7. It solves the Poisson equation via FFT to recover the unwrapped phase.
    
    The returned unwrapped phase is masked to have the same geometry as the input array.
    
    Parameters
    ----------
    wrapped : 2D numpy array
        Wrapped phase in radians (ideally in the interval [-π, π]; data may span a wider range).
        The array may contain NaNs.
    window : int, optional
        Size of the local window used for quality filtering (default is 5).
    
    Returns
    -------
    unwrapped : 2D numpy array
        Unwrapped phase in radians. This continuous phase field can be converted to physical
        displacement (e.g., using d = unwrapped_phase * (λ / (4π))) and is masked to the input geometry.
    
    References
    ----------
    Lu, Y., Wang, X., & Zhang, X. (2007). Weighted least-squares phase unwrapping algorithm based on 
    derivative variance correlation map. Optik, 118(2), 62–66.
    """
    # -------------------------------------------------------------------------
    # 1. Ensure no NaNs are present.
    # NaNs will disrupt arithmetic and FFT computations.
    # Replace them with the mean of valid values so that further computations are not affected.
    if np.isnan(wrapped).any():
        fill_value = np.nanmean(wrapped)  # Compute mean of all valid phase values.
        wrapped = np.where(np.isnan(wrapped), fill_value, wrapped)  # Replace NaNs.

    # Get the dimensions of the input wrapped phase image.
    M, N = wrapped.shape

    # -------------------------------------------------------------------------
    # 1.5. Quality Filtering Step (Pre-Processing)
    # Compute a quality map by evaluating the local variance of the phase gradient.
    # High variance indicates potential discontinuities or noise.
    # This step uses a small local window (controlled by the 'window' argument).
    # First, compute finite differences for quality estimation.
    dx_q = np.diff(wrapped, axis=1)  # Horizontal phase difference; shape (M, N-1)
    dy_q = np.diff(wrapped, axis=0)  # Vertical phase difference; shape (M-1, N)
    # Compute gradient magnitude on the overlapping region (size: (M-1, N-1)).
    # Use the first M-1 rows of dx_q and first N-1 columns of dy_q.
    grad_mag = np.sqrt(dx_q[:-1, :]**2 + dy_q[:, :-1]**2)
    
    # Compute the local mean and the local mean of squared gradients using a uniform filter.
    local_mean = uniform_filter(grad_mag, size=window)
    local_mean_sq = uniform_filter(grad_mag**2, size=window)
    #local_variance = local_mean_sq - local_mean**2

    # Define an adaptive threshold based on the median and standard deviation of grad_mag.
    threshold = np.median(grad_mag) + np.std(grad_mag)

    # Create a reliability mask: reliable pixels have local gradient magnitude below the threshold.
    reliable_mask = grad_mag < threshold
    # Pad the mask back to the full image size.
    mask_full = np.pad(reliable_mask, ((1, 0), (1, 0)), mode='edge')

    # Create a smoothed version of the wrapped phase using a circular (complex) average.
    # Convert the wrapped phase to its complex representation.
    wrapped_complex = np.exp(1j * wrapped)
    smoothed_real = uniform_filter(wrapped_complex.real, size=window)
    smoothed_imag = uniform_filter(wrapped_complex.imag, size=window)
    smoothed = np.angle(smoothed_real + 1j * smoothed_imag)

    # Replace unreliable pixels (where mask is False) with the locally smoothed phase value.
    wrapped_filtered = np.where(mask_full, wrapped, smoothed)
    
    # NOTE: The fill value replacement and quality filtering may affect phase values at the edges.
    # Ensure that the geometry of the input is preserved.
    wrapped = wrapped_filtered

    # -------------------------------------------------------------------------
    # 2. Compute Finite Differences (Phase Gradients)
    # Compute differences between adjacent pixels to capture local phase changes.
    dx = np.diff(wrapped, axis=1)  # Horizontal differences; shape (M, N-1)
    dy = np.diff(wrapped, axis=0)  # Vertical differences; shape (M-1, N)
    # NOTE: These differences represent local phase changes, not physical deformation.

    # -------------------------------------------------------------------------
    # 3. Wrap the Finite Differences to the Interval [-π, π]
    # Mapping raw differences to the principal value ensures that we work with the smallest angular differences.
    dx = np.angle(np.exp(1j * dx))
    dy = np.angle(np.exp(1j * dy))

    # -------------------------------------------------------------------------
    # 4. Expand the Computed Differences Back to the Full Image Size
    # The np.diff function reduces the dimensions by one.
    # Create full-size arrays (M, N) and insert the computed differences in their valid positions.
    dx_full = np.zeros((M, N), dtype=wrapped.dtype)
    dy_full = np.zeros((M, N), dtype=wrapped.dtype)
    dx_full[:, :-1] = dx  # Fill all but the last column.
    dy_full[:-1, :] = dy  # Fill all but the last row.

    # -------------------------------------------------------------------------
    # 5. Build the Divergence (Right-Hand Side of the Poisson Equation)
    # The divergence represents the net "flow" of phase differences out of each pixel.
    # Compute it by taking differences of the full-sized finite differences.
    rhs = np.zeros((M, N), dtype=wrapped.dtype)
    # Compute x divergence:
    rhs[:, 0] = dx_full[:, 0]                   # First column (no left neighbor).
    rhs[:, 1:] = dx_full[:, 1:] - dx_full[:, :-1] # Differences along columns.
    # Compute y divergence:
    rhs[0, :] += dy_full[0, :]                  # First row (no upper neighbor).
    rhs[1:, :] += dy_full[1:, :] - dy_full[:-1, :]# Differences along rows.
    # This divergence integrates the differences between measured and true phase gradients.

    # -------------------------------------------------------------------------
    # 6. Solve the Poisson Equation Using FFT
    # The Poisson equation in this context is: ∇² φ_unwrapped = divergence.
    # Taking the FFT converts the Laplacian to a multiplication in the frequency domain.
    fft_rhs = np.fft.fft2(rhs)  # Compute the 2D FFT of the divergence field.
    
    # Generate frequency grids for x and y.
    x_freq = np.fft.fftfreq(N) * 2 * np.pi  # Frequencies in cycles per pixel (converted to radians).
    y_freq = np.fft.fftfreq(M) * 2 * np.pi
    # Create a 2D meshgrid of frequencies (using 'ij' indexing to match array dimensions).
    Y_freq, X_freq = np.meshgrid(y_freq, x_freq, indexing='ij')
    
    # Compute the eigenvalues of the discrete Laplacian operator for a pixel grid.
    # These are given by: 2*cos(X) + 2*cos(Y) - 4, rearranged as (2*cos(X)-2) + (2*cos(Y)-2).
    denom = (2 * np.cos(X_freq) - 2) + (2 * np.cos(Y_freq) - 2)
    # Set the zero-frequency (DC) component to 1 to avoid division by zero.
    denom[0, 0] = 1.0
    
    # Divide the FFT of the divergence by the eigenvalue spectrum to obtain the FFT of the unwrapped phase.
    fft_phi = fft_rhs / denom
    # Set the DC component (overall mean) to 0, since absolute phase is ambiguous.
    fft_phi[0, 0] = 0.0
    
    # Invert the FFT to transform back to the spatial domain, yielding the unwrapped phase.
    # Take only the real part because the imaginary part is negligible (due to numerical noise).
    unwrapped = np.real(np.fft.ifft2(fft_phi))
    
    # -------------------------------------------------------------------------
    # 7. Mask the Unwrapped Phase to the Original Geometry
    # Ensure that the output unwrapped phase has the same shape as the original input array.
    unwrapped = unwrapped[:M, :N]
    
    return unwrapped

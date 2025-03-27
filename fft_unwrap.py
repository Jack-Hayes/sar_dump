import numpy as np
from scipy.ndimage import uniform_filter

def unwrap_phase_fft(wrapped, window=5, threshold_factor=1.5):
    """
    Unwrap a 2D wrapped phase image using an FFT-based least-squares approach 
    with an adaptive, quality filtering pre-step.
    
    The underlying concept is that the unwrapped phase φ_unwrapped at each pixel (i, j)
    can be expressed as:
    
       φ_unwrapped(i, j) = φ_wrapped(i, j) + 2π · (integer offset)
    
    The function performs the following main steps:
    
      1. **NaN Replacement:** 
         Replace any NaNs in the wrapped phase image with the mean of the valid values.
      
      2. **Adaptive Quality Filtering:**
         - Compute finite differences along the horizontal and vertical directions 
           (dx_q, dy_q) to estimate local phase gradients.
         - Calculate the gradient magnitude over the overlapping region.
         - Using a uniform filter, compute the local mean and local mean of squared gradients 
           to derive the local variance and local standard deviation.
         - Build a spatially adaptive threshold at each pixel as:
               
               adaptive_threshold = local_mean + threshold_factor * local_std
               
         - Construct a reliability mask where pixels with a gradient magnitude below the 
           adaptive threshold are considered reliable.
         - Replace unreliable pixels (those where the mask is False) with a locally 
           smoothed phase value. The smoothing is performed using a circular (complex) average 
           computed from the complex representation of the wrapped phase.
      
      3. **Finite Difference Calculation:**
         - Compute the phase gradients (dx, dy) from the filtered (and quality-enhanced) 
           phase.
         - Wrap these gradients to the principal value interval [-π, π].
      
      4. **Expansion and Divergence Calculation:**
         - Expand the computed gradients to full image size.
         - Compute the divergence of the gradient field, which forms the right-hand side (RHS)
           of the Poisson equation.
      
      5. **Poisson Equation Solving:**
         - Solve the discrete Poisson equation (∇² φ = RHS) using FFT. In the frequency domain, 
           the Laplacian becomes a multiplicative factor that is inverted.
         - Set the zero-frequency (DC) component to 0 to remove the ambiguity in absolute phase.
      
      6. **Output:**
         - The resulting unwrapped phase image is returned with the same geometry as the input.
    
    Parameters
    ----------
    wrapped : 2D numpy array
        Wrapped phase image in radians. Although ideally the phase is in [-π, π],
        the data may span a wider range and may contain NaNs.
    window : int, optional
        Size of the local window (in pixels) used for computing local statistics for 
        quality filtering. Default is 5.
    threshold_factor : float, optional
        Multiplicative factor for the local standard deviation when computing the 
        adaptive threshold. A higher value makes the threshold more lenient. Default is 1.5.
    
    Returns
    -------
    unwrapped : 2D numpy array
        The unwrapped phase image in radians. This continuous phase field can be 
        further used (e.g., multiplied by a scaling factor) to convert into physical 
        displacement or other quantities. The geometry of the output is the same as 
        the input.
    
    References
    ----------
    Lu, Y., Wang, X., & Zhang, X. (2007). Weighted least-squares phase unwrapping algorithm 
    based on derivative variance correlation map. Optik, 118(2), 62–66.
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
    local_variance = local_mean_sq - local_mean**2
    local_std = np.sqrt(np.maximum(local_variance, 0))

    # Build a spatially adaptive threshold:
    # For each pixel, threshold = local_mean + threshold_factor * local_std.
    adaptive_threshold = local_mean + threshold_factor * local_std

    # Create a reliability mask: reliable pixels have local gradient magnitude below the threshold.
    reliable_mask = grad_mag < adaptive_threshold
    # Pad the mask back to the full image size.
    mask_full = np.pad(reliable_mask, ((1, 0), (1, 0)), mode='edge')

    # Create a smoothed version of the wrapped phase using a circular (complex) average.
    # Convert the wrapped phase to its complex representation.
    wrapped_complex = np.exp(1j * wrapped)
    smoothed_real = uniform_filter(wrapped_complex.real, size=window)
    smoothed_imag = uniform_filter(wrapped_complex.imag, size=window)
    smoothed_phase = np.angle(smoothed_real + 1j * smoothed_imag)

    # Replace unreliable pixels (where mask is False) with the locally smoothed phase value.
    wrapped_filtered = np.where(mask_full, wrapped, smoothed_phase)
    
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

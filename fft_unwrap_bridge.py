import numpy as np
import time
from scipy.ndimage import uniform_filter, label, binary_erosion

# NOTE: I'm currently working on ways to dynamically assign
# bridge window and region size parameters based on the
# image size, resolution, and fringe characteristics
# (but will still allow users to define their own params for these)
# TODO: The stochastic nature of this function returns wildly different
#       results for the identical data inputs... need to figure out
#       why and make it at least somewhat consistent...
def unwrap_phase_fft_bridge(wrapped, window=5, threshold_factor=1.5,
                             min_region_size=100, bridge_window=16, verbose=True):
    """
    Unwrap a 2D wrapped interferometric phase image with fast Poisson-based least-squares
    plus a lightweight bridging correction. This version:

      - **Enlarges sampling window** (bridge_window=16) to exceed the characteristic
        decorrelation scale (e.g., >100 m for Kīlauea 2018 eruption scene) so that
        phase-difference estimates average over coherent patches spanning lava flows.
      - **Trimmed-mean thresholds** ([15, 85] percentile) to reflect heavy
        noise/outlier tails induced by decorrelated lava, crater walls, and fumaroles.
      - **Linear ramp removal/add-back** on the largest reliable region to decouple
        long-wavelength volcanic deformation signals (meters/day) from integer-cycle offsets.

    **These settings are not universal**—they must be tuned to the site’s coherence field
    and deformation amplitude. For example, on Kīlauea’s rapid, multi-meter/day
    inflation/deflation in 2018, bridge_window=16 and a tight trim prevented
    over- or under-shifting across fresh lava flows, whereas smaller windows or
    looser trims failed to span decorrelated patches.

    Parameters
    ----------
    wrapped : 2D numpy array
        Wrapped-phase in radians. May span outside [-π, π] and contain NaNs.
    window : int
        Local window (pixels) for adaptive quality filter. Default 5.
    threshold_factor : float
        Multiplier for local std in quality threshold. Default 1.5.
    min_region_size : int
        Minimum pixels in a region to consider for ramp removal. Default 100.
    bridge_window : int
        Half-size (16 px) of sampling window for trimmed-mean offset.
    verbose : bool
        Print timing diagnostics if True.

    Returns
    -------
    unwrapped : 2D numpy array
        Continuous unwrapped-phase, corrected for large-scale jumps.
    """
    # start timer for total runtime
    t0 = time.perf_counter()

    # ------------------------------------------------------------------------
    # 1. NaN Replacement: interpolate NaNs with Gaussian noise matching data stats
    if np.isnan(wrapped).any():                            # detect any NaNs
        valid = wrapped[np.isfinite(wrapped)]              # extract finite pixels
        mu, sigma = valid.mean(), valid.std()              # compute mean & std
        wrapped = np.where(np.isnan(wrapped),             # replace NaNs
                           np.random.normal(mu, sigma, wrapped.shape),
                           wrapped)
    M, N = wrapped.shape                                  # image dimensions
    t1 = time.perf_counter()
    if verbose:
        print(f"NaN replacement: {t1-t0:.3f}s")

    # ------------------------------------------------------------------------
    # 2. Adaptive Quality Filtering: smooth pixels with high gradient variance
    dx_q = np.diff(wrapped, axis=1)                       # horizontal diff
    dy_q = np.diff(wrapped, axis=0)                       # vertical diff
    grad_mag = np.hypot(dx_q[:-1, :], dy_q[:, :-1])       # compute gradient magnitude
    local_mean = uniform_filter(grad_mag, window)         # local mean of gradients
    local_var = uniform_filter(grad_mag**2, window) - local_mean**2  # var = E[X²]-E[X]²
    local_std = np.sqrt(np.clip(local_var, 0, None))      # standard deviation
    mask_full = np.pad(                                  
        grad_mag < (local_mean + threshold_factor*local_std),
        ((1, 0), (1, 0)), mode='edge'                   # pad to full image
    )
    c = np.exp(1j * wrapped)                             # complex representation
    smooth = np.angle(                                   
        uniform_filter(c.real, window)                  
        + 1j * uniform_filter(c.imag, window)           # circular average
    )
    wrapped = np.where(mask_full, wrapped, smooth)       # replace unreliable
    t2 = time.perf_counter()
    if verbose:
        print(f"Quality filtering: {t2-t1:.3f}s")

    # ------------------------------------------------------------------------
    # 3. Poisson Solve: least-squares unwrapping via FFT
    dx = np.angle(np.exp(1j * np.diff(wrapped, axis=1)))  # wrapped horizontal gradients
    dy = np.angle(np.exp(1j * np.diff(wrapped, axis=0)))  # wrapped vertical gradients
    rhs = np.zeros((M, N))                                # initialize divergence
    rhs[:, 0] = dx[:, 0]                                  # x-divergence first col
    rhs[:, 1:] = dx - np.pad(dx[:, :-1], ((0,0),(1,0)), 'constant')  # other cols
    rhs[0, :] += dy[0, :]                                 # y-div first row
    rhs[1:, :] += dy - np.pad(dy[:-1, :], ((1,0),(0,0)), 'constant')  # other rows
    fft_rhs = np.fft.rfft2(rhs)                           # FFT of divergence
    xf = np.fft.rfftfreq(N) * 2*np.pi                     # freq grid x
    yf = np.fft.fftfreq(M) * 2*np.pi                      # freq grid y
    Yf, Xf = np.meshgrid(yf, xf, indexing='ij')          # 2D freq mesh
    denom = (2*np.cos(Xf)-2) + (2*np.cos(Yf)-2)          # Laplacian eigenvalues
    denom[0,0] = 1.0                                     # avoid zero-division
    phi = fft_rhs / denom                                 # solve in freq domain
    phi[0,0] = 0.0                                       # zero DC component
    unwrapped = np.real(np.fft.irfft2(phi, s=(M, N)))    # invert FFT to spatial
    t3 = time.perf_counter()
    if verbose:
        print(f"Poisson solve: {t3-t2:.3f}s")

    # ------------------------------------------------------------------------
    # 4. Linear Ramp Removal: remove long-wavelength trend on main region
    labels, _ = label(mask_full)                         # label connected regions
    sizes = np.bincount(labels.ravel())                  # pixel counts per region
    keep = np.where(sizes >= min_region_size)[0]         # select large regions
    keep = keep[keep != 0]                               # drop background label
    if len(keep) == 0:
        return unwrapped                                  # no region to correct
    main_label = keep[np.argmax(sizes[keep])]            # pick largest region
    ys, xs = np.where(labels == main_label)              # coords of main region
    A = np.vstack([xs, ys, np.ones_like(xs)]).T         # design matrix for plane
    coeffs, *_ = np.linalg.lstsq(A, unwrapped[ys, xs], rcond=None)  # fit plane
    ramp = (coeffs[0]*np.arange(N)[None, :]
            + coeffs[1]*np.arange(M)[:, None]
            + coeffs[2])                              # ramp surface
    unwrapped0 = unwrapped - ramp                        # subtract ramp

    # ------------------------------------------------------------------------
    # 5. Fast Bridging Correction: trimmed-mean over large window
    labels2, num = label(mask_full)                      # re-label for bridging
    mask2 = binary_erosion(np.isin(labels2, keep))       # erode region boundaries
    regions, num = label(mask2)                          # label eroded mask
    if num <= 1:
        t4 = time.perf_counter()
        if verbose:
            print(f"Bridging skipped: {t4-t3:.3f}s")
        unwrapped += ramp                                 # re-add ramp
        return unwrapped

    # centroid estimation for each region
    centroids = []
    for lab in keep:
        ys_lab, xs_lab = np.where(labels2 == lab)        # coords per label
        centroids.append(((ys_lab.min()+ys_lab.max())/2,
                          (xs_lab.min()+xs_lab.max())/2))
    centroids = np.array(centroids)                       # array of centroids

    # trimmed-mean estimator to reject tails
    def tmean(arr):
        flat = arr.ravel()
        lo, hi = np.percentile(flat, [15, 85])           # cut worst 15%
        mid = flat[(flat >= lo) & (flat <= hi)]
        return mid.mean() if mid.size > 0 else np.median(flat)

    largest_idx = np.argmax(sizes[keep])                  # index of main region
    yL, xL = map(int, centroids[largest_idx])            # main centroid
    w = bridge_window                                     # window half-size
    base_win = unwrapped0[max(yL-w,0):yL+w+1,             # sample base window
                           max(xL-w,0):xL+w+1]
    base_val = tmean(base_win)                            # robust base phase

    # apply integer-cycle shifts to each region
    for idx, (y, x) in enumerate(centroids):
        if idx == largest_idx:
            continue                                     # skip base region
        y, x = map(int, (y, x))                          # region centroid
        win = unwrapped0[max(y-w,0):y+w+1,                # local sample window
                        max(x-w,0):x+w+1]
        delta = base_val - tmean(win)                    # phase difference
        k = int(np.round(delta/(2*np.pi)))              # integer cycles to shift
        unwrapped[labels2 == keep[idx]] += k*2*np.pi    # apply shift
    t4 = time.perf_counter()
    if verbose:
        print(f"Bridging: {t4-t3:.3f}s")

    # ------------------------------------------------------------------------
    # 6. Add the ramp back to restore long-wavelength deformation
    unwrapped += ramp                                     # restore removed ramp
    if verbose:
        print(f"Total: {t4-t0:.3f}s")
    return unwrapped


# NOTE: this takes way too long (22+ mins compared to the sub-minute fft_unwrap.py for our use case)
# updated 05/06/2025
# import numpy as np
# from scipy.ndimage import uniform_filter, label, binary_erosion


# def unwrap_phase_fft_bridge(wrapped, window=5, threshold_factor=1.5,
#                             min_region_size=100, bridge_window=3):
#     """
#     Unwrap a 2D wrapped phase image using an FFT-based least-squares approach
#     with an adaptive, quality filtering pre-step, plus a fast bridging of reliable regions
#     to correct large phase jumps between decorrelated areas.

#     The function proceeds in the following stages:
#       1. **NaN Replacement**: Replace any NaNs with Gaussian noise matching data distribution.
#       2. **Adaptive Quality Filtering**:
#          - Compute local phase gradients (dx_q, dy_q).
#          - Derive local gradient magnitude and compute adaptive thresholds using local mean and std.
#          - Build a reliability mask and smooth unreliable pixels via complex averaging.
#       3. **Finite Difference Calculation**: Compute wrapped gradients dx, dy in [-π, π].
#       4. **Poisson Solver**: Build divergence (RHS) and solve ∇²φ = RHS via FFT for initial unwrapping.
#       5. **Bridging of Reliable Regions**:
#          - Identify connected reliable regions from quality mask.
#          - Discard small regions and erode boundaries to avoid edge effects.
#          - Compute centroids of remaining regions.
#          - Construct a minimal spanning tree (MST) among centroids (Prim's algorithm).
#          - Estimate integer-cycle offsets between region pairs by sampling median phase in small windows.
#          - Apply 2π·k shifts to align all regions and remove large jumps.

#     Parameters
#     ----------
#     wrapped : ndarray
#         2D wrapped-phase array (radians). Contains NaNs and values outside [-π, π].
#     window : int
#         Local window size for computing quality filter statistics (default 5 pixels).
#     threshold_factor : float
#         Multiplier for local std when computing adaptive threshold (default 1.5).
#     min_region_size : int
#         Minimum pixel count for a reliable region to be considered in bridging (default 100).
#     bridge_window : int
#         Half-window size (in pixels) around centroids for median-phase sampling (default 3).

#     Returns
#     -------
#     unwrapped : ndarray
#         Fully unwrapped phase image in radians, with large jumps corrected.
#     """
#     # -------------------------------------------------------------------------
#     # 1. NaN Replacement
#     # Replace NaNs with synthetic phase values drawn from a Gaussian distribution
#     # matching the mean and std dev of valid wrapped-phase values.
#     if np.isnan(wrapped).any():
#         valid = wrapped[np.isfinite(wrapped)]               # extract finite pixels
#         mu, sigma = valid.mean(), valid.std()              # compute mean, std
#         noise = np.random.normal(mu, sigma, wrapped.shape) # generate Gaussian noise
#         wrapped = np.where(np.isnan(wrapped), noise, wrapped)  # fill NaNs with noise

#     # Get image dimensions
#     M, N = wrapped.shape

#     # -------------------------------------------------------------------------
#     # 2. Adaptive Quality Filtering
#     # Compute raw finite differences along x and y for quality estimation.
#     dx_q = np.diff(wrapped, axis=1)  # horizontal differences, shape (M, N-1)
#     dy_q = np.diff(wrapped, axis=0)  # vertical differences, shape (M-1, N)

#     # Compute gradient magnitude over overlapping region (M-1, N-1)
#     grad_mag = np.sqrt(dx_q[:-1, :]**2 + dy_q[:, :-1]**2)

#     # Compute local statistics via uniform_filter
#     local_mean = uniform_filter(grad_mag, size=window)       # mean of gradient magnitude
#     local_mean_sq = uniform_filter(grad_mag**2, size=window) # mean of squared gradients
#     local_variance = local_mean_sq - local_mean**2           # variance = E[X^2] - E[X]^2
#     local_std = np.sqrt(np.clip(local_variance, 0, None))    # avoid negative variance

#     # Build adaptive threshold per pixel
#     adaptive_thresh = local_mean + threshold_factor * local_std

#     # Reliability mask: True where gradient magnitude below threshold
#     reliable = grad_mag < adaptive_thresh
#     # Pad mask to full image size, replicating edge values
#     mask_full = np.pad(reliable, ((1, 0), (1, 0)), mode='edge')

#     # Smooth unreliable pixels by circular (complex) averaging
#     wrapped_c = np.exp(1j * wrapped)                         # complex representation
#     smooth_real = uniform_filter(wrapped_c.real, size=window)
#     smooth_imag = uniform_filter(wrapped_c.imag, size=window)
#     smooth_phase = np.angle(smooth_real + 1j * smooth_imag)  # averaged phase

#     # Replace unreliable pixels with smoothed phase
#     wrapped = np.where(mask_full, wrapped, smooth_phase)

#     # -------------------------------------------------------------------------
#     # 3. Compute Finite Differences & Wrap to [-π, π]
#     dx = np.diff(wrapped, axis=1)     # new wrapped-phase differences
#     dy = np.diff(wrapped, axis=0)
#     dx = np.angle(np.exp(1j * dx))    # wrap to principal interval
#     dy = np.angle(np.exp(1j * dy))

#     # -------------------------------------------------------------------------
#     # 4. Expand Diffs & Build Divergence (RHS of Poisson)
#     dx_full = np.zeros((M, N), dtype=wrapped.dtype)
#     dy_full = np.zeros((M, N), dtype=wrapped.dtype)
#     dx_full[:, :-1] = dx             # fill all but last column
#     dy_full[:-1, :] = dy             # fill all but last row

#     # Initialize divergence RHS
#     rhs = np.zeros((M, N), dtype=wrapped.dtype)
#     # x-divergence: first column + differences along columns
#     rhs[:, 0] = dx_full[:, 0]
#     rhs[:, 1:] = dx_full[:, 1:] - dx_full[:, :-1]
#     # y-divergence: first row + differences along rows
#     rhs[0, :] += dy_full[0, :]
#     rhs[1:, :] += dy_full[1:, :] - dy_full[:-1, :]

#     # Solve Poisson ∇²φ = rhs via FFT
#     fft_rhs = np.fft.fft2(rhs)
#     # Frequency grids (radians per pixel)
#     x_freq = np.fft.fftfreq(N) * 2 * np.pi
#     y_freq = np.fft.fftfreq(M) * 2 * np.pi
#     Yf, Xf = np.meshgrid(y_freq, x_freq, indexing='ij')

#     # Discrete Laplacian eigenvalues: (2cos(kx)-2)+(2cos(ky)-2)
#     denom = (2 * np.cos(Xf) - 2) + (2 * np.cos(Yf) - 2)
#     denom[0, 0] = 1.0  # avoid division by zero at DC

#     # Compute unwrapped-phase FFT and zero DC component
#     fft_phi = fft_rhs / denom
#     fft_phi[0, 0] = 0.0

#     # Inverse FFT to spatial domain, keep real part
#     unwrapped = np.real(np.fft.ifft2(fft_phi))

#     # -------------------------------------------------------------------------
#     # 5. Bridging of Reliable Regions (to correct large jumps)
#     # Create binary mask of reliable regions from pre-filter mask
#     bw = mask_full.copy().astype(bool)

#     # Label connected components in the reliability mask
#     regions, num_regions = label(bw)

#     # Discard small regions below min_region_size
#     keep_labels = []
#     for lab in range(1, num_regions + 1):
#         coords = np.argwhere(regions == lab)
#         if coords.shape[0] >= min_region_size:
#             keep_labels.append(lab)

#     # Build mask of kept regions
#     mask_regs = np.isin(regions, keep_labels)
#     # Erode boundaries to avoid edge artifacts
#     mask_regs = binary_erosion(mask_regs, iterations=1)
#     # Relabel after erosion
#     regions, num_regions = label(mask_regs)

#     # If only one region remains, no bridging needed
#     if num_regions <= 1:
#         return unwrapped

#     # Compute centroids (row, col) of each reliable region
#     centroids = []
#     for lab in range(1, num_regions + 1):
#         ys, xs = np.where(regions == lab)
#         centroids.append((ys.mean(), xs.mean()))
#     centroids = np.array(centroids)

#     # Build MST connecting region centroids via Prim's algorithm
#     used = {0}
#     edges = []
#     # Precompute pairwise Euclidean distances
#     dists = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :])**2, axis=-1))
#     while len(used) < num_regions:
#         min_d, i_min, j_min = np.inf, None, None
#         for i in used:
#             for j in range(num_regions):
#                 if j not in used and dists[i, j] < min_d:
#                     min_d, i_min, j_min = dists[i, j], i, j
#         edges.append((i_min, j_min))
#         used.add(j_min)

#     # For each MST edge, compute integer-phase offset and apply shift
#     for i, j in edges:
#         y1, x1 = map(int, centroids[i])
#         y2, x2 = map(int, centroids[j])
#         w = bridge_window
#         # Extract small windows around centroids
#         win1 = unwrapped[max(y1-w, 0):y1+w+1, max(x1-w, 0):x1+w+1]
#         win2 = unwrapped[max(y2-w, 0):y2+w+1, max(x2-w, 0):x2+w+1]
#         # Median-phase difference between windows
#         delta = np.median(win1) - np.median(win2)
#         # Compute integer cycles k to bring delta within [-π, π]
#         k = np.round(delta / (2 * np.pi)).astype(int)
#         # Shift region j by k * 2π to align with region i
#         unwrapped[regions == (j + 1)] += k * 2 * np.pi

#     # Return final unwrapped and bridged phase
#     return unwrapped

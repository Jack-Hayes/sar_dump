import numpy as np
import scipy.ndimage as ndi

def make_grid(n=1000):
    """Return coordinate arrays X, Y over [-1,1]×[-1,1]."""
    axis = np.linspace(-1, 1, n)
    return np.meshgrid(axis, axis)

def generate_mogi(n=1000, center=(0,0), depth=0.2, deltaV=1.0):
    """Spheroidal/Mogi source uplift pattern (radial, bell-shaped)."""
    X, Y = make_grid(n)
    r = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return deltaV * depth / (r**2 + depth**2)**1.5

def generate_tilt(n=1000, slope_x=0.1, slope_y=0.2):
    """Linear planar ramp."""
    X, Y = make_grid(n)
    return slope_x * X + slope_y * Y

def generate_dislocation(n=1000, fault_angle=0.0, jump=1.0):
    """Edge dislocation: a step across a line at given angle (radians)."""
    X, Y = make_grid(n)
    Xr = X * np.cos(fault_angle) + Y * np.sin(fault_angle)
    return jump * np.sign(Xr)

def combine_sources(n, source_list, weight_list=None):
    """Linear combination of multiple unwrapped sources."""
    if weight_list is None:
        weight_list = [1.0] * len(source_list)
    phi = np.zeros((n, n))
    for w, src in zip(weight_list, source_list):
        phi += w * src
    return phi

def wrap_phase(phi):
    """Ideal wrapping to [-pi, pi)."""
    return np.angle(np.exp(1j * phi))

def add_gaussian_noise(wrapped_phase, sigma=0.1):
    """Add Gaussian noise to complex interferogram before re-wrapping."""
    c = np.exp(1j * wrapped_phase)
    noise = (np.random.normal(scale=sigma, size=c.shape) +
             1j * np.random.normal(scale=sigma, size=c.shape))
    return np.angle(c + noise)

def add_speckle(phi, coherence=0.8):
    """Simulate speckle by generating two speckled SAR images."""
    amp1 = np.sqrt(-2 * np.log(np.random.rand(*phi.shape)))
    amp2 = np.sqrt(-2 * np.log(np.random.rand(*phi.shape)))
    img1 = amp1 * np.exp(1j * np.random.rand(*phi.shape) * 2 * np.pi)
    img2 = amp2 * np.exp(1j * (phi + np.random.rand(*phi.shape) * 2 * np.pi))
    ifg = img1 * np.conj(img2)
    mag = coherence * np.abs(ifg) + (1 - coherence) * np.mean(np.abs(ifg))
    return np.angle(ifg / np.abs(ifg) * mag)

def add_random_nans(wrapped_phase, percent=0.1):
    """Randomly scatter NaNs over a given percentage of pixels."""
    mask = np.random.rand(*wrapped_phase.shape) < percent
    wp = wrapped_phase.copy()
    wp[mask] = np.nan
    return wp

def add_patch_nans(wrapped_phase, num_patches=3, max_radius=100):
    """Insert contiguous circular NaN patches."""
    wp = wrapped_phase.copy()
    n = wp.shape[0]
    for _ in range(num_patches):
        cx, cy = np.random.randint(0, n, size=2)
        r = np.random.randint(20, max_radius)
        Y, X = np.ogrid[:n, :n]
        mask = (X - cx)**2 + (Y - cy)**2 <= r*r
        wp[mask] = np.nan
    return wp

def add_atmospheric_noise(phi, scale=0.05, smooth_sigma=50):
    """Add smooth, spatially correlated 'atmospheric' phase."""
    noise = np.random.randn(*phi.shape) * scale
    return phi + ndi.gaussian_filter(noise, sigma=smooth_sigma)

# Non-Gaussian, spatially correlated speckle + thermal noise simulation
def add_non_gaussian_noise(phi, thermal_sigma=0.05, speckle_coh=0.8, smooth_sigma=30):
    """Combine thermal noise (Gaussian) and speckle (spatially correlated) phases."""
    # Thermal noise: additive Gaussian phase noise
    thermal_noise = np.random.randn(*phi.shape) * thermal_sigma

    # Speckle noise via two independent SAR emulations
    amp1 = np.sqrt(-2 * np.log(np.random.rand(*phi.shape)))
    amp2 = np.sqrt(-2 * np.log(np.random.rand(*phi.shape)))
    ph1 = np.random.rand(*phi.shape) * 2 * np.pi
    ph2 = phi + np.random.rand(*phi.shape) * 2 * np.pi
    img1 = amp1 * np.exp(1j * ph1)
    img2 = amp2 * np.exp(1j * ph2)
    ifg = img1 * np.conj(img2)
    mag = speckle_coh * np.abs(ifg) + (1 - speckle_coh) * np.mean(np.abs(ifg))
    speckle_phase = np.angle(ifg * mag / np.abs(ifg))

    # Combine and smooth to induce spatial correlation
    combined = phi + thermal_noise + speckle_phase
    return np.angle(np.exp(1j * ndi.gaussian_filter(combined, sigma=smooth_sigma)))
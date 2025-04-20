import numpy as np

def mf_basic(data):
    """
    Matched Filter Algorithm

    Performs a matched filter operation. Expects the following attributes on `data`:
      - deltaF: frequency step size (Hz)
      - minF:  array of start frequencies per pulse (Hz), shape (Np,)
      - x_mat, y_mat, z_mat: 2D arrays of pixel coordinates (m)
      - AntX, AntY, AntZ: arrays of sensor coords at each pulse (m), shape (Np,)
      - R0: array of range to scene center for each pulse (m), shape (Np,)
      - phdata: 2D complex array of phase history data, shape (K, Np)

    Returns:
      - data.im_final: 2D complex image array, same shape as x_mat
    """
    # Define speed of light (m/s)
    c = 299792458

    # Determine the size of the phase history data
    data.K = data.phdata.shape[0]  # number of frequency bins per pulse
    data.Np = data.phdata.shape[1]  # number of pulses

    # Determine the azimuth angles of the image pulses (radians)
    data.AntAz = np.unwrap(np.arctan2(data.AntY, data.AntX))

    # Determine the average azimuth angle step size (radians)
    data.deltaAz = np.abs(np.mean(np.diff(data.AntAz)))

    # Determine the total azimuth angle of the aperture (radians)
    data.totalAz = np.max(data.AntAz) - np.min(data.AntAz)

    # Determine the maximum scene size of the image (m)
    data.maxWr = c / (2 * data.deltaF)
    data.maxWx = c / (2 * data.deltaAz * np.mean(data.minF))

    # Determine the resolution of the image (m)
    data.dr = c / (2 * data.deltaF * data.K)
    data.dx = c / (2 * data.totalAz * np.mean(data.minF))

    # Display maximum scene size and resolution
    print(f"Maximum Scene Size: {data.maxWr:.2f} m range, {data.maxWx:.2f} m cross-range")
    print(f"Resolution: {data.dr:.2f} m range, {data.dx:.2f} m cross-range")

    # Initialize the image with all zero values
    data.im_final = np.zeros_like(data.x_mat, dtype=np.complex128)

    # Set up a vector to keep execution times for each pulse (sec)
    t = np.zeros(data.Np)

    # Loop through every pulse
    for ii in range(data.Np):
        # Display status of the imaging process
        if ii > 0:
            t_sofar = np.sum(t[:ii])
            t_est = (t_sofar * data.Np / ii - t_sofar) / 60
            print(f"Pulse {ii+1} of {data.Np}, {t_est:.2f} minutes remaining")
        else:
            print(f"Pulse 1 of {data.Np}")
        start = np.nan; import time; start = time.time()  # start timer

        # Calculate differential range for each pixel in the image (m)
        dR = np.sqrt((data.AntX[ii] - data.x_mat)**2 +
                     (data.AntY[ii] - data.y_mat)**2 +
                     (data.AntZ[ii] - data.z_mat)**2) - data.R0[ii]

        # Calculate the frequency of each sample in the pulse (Hz)
        freq = data.minF[ii] + np.arange(data.K) * data.deltaF

        # Perform the Matched Filter operation
        for jj in range(data.K):
            data.im_final += data.phdata[jj, ii] * np.exp(1j * 4 * np.pi * freq[jj] / c * dR)

        # Determine the execution time for this pulse
        t[ii] = time.time() - start

    return data

def bp_basic(data):
    """
    Backprojection Algorithm

    Performs a basic backprojection operation. Expects the following attributes on `data`:
      - Nfft: size of FFT for range profile
      - deltaF: frequency step size (Hz)
      - minF:  array of start frequencies per pulse (Hz), shape (Np,)
      - x_mat, y_mat, z_mat: 2D arrays of pixel coordinates (m)
      - AntX, AntY, AntZ: arrays of sensor coords at each pulse (m), shape (Np,)
      - R0: array of range to scene center for each pulse (m), shape (Np,)
      - phdata: 2D complex array of phase history data, shape (K, Np)

    Returns:
      - data.im_final: 2D complex image array, same shape as x_mat
    """
    # Define speed of light (m/s)
    c = 299792458

    # Determine the size of the phase history data
    data.K = data.phdata.shape[0]  # number of frequency bins per pulse
    data.Np = data.phdata.shape[1]  # number of pulses

    # Determine the azimuth angles of the image pulses (radians)
    data.AntAz = np.unwrap(np.arctan2(data.AntY, data.AntX))

    # Determine the average azimuth angle step size (radians)
    data.deltaAz = np.abs(np.mean(np.diff(data.AntAz)))

    # Determine the total azimuth angle of the aperture (radians)
    data.totalAz = np.max(data.AntAz) - np.min(data.AntAz)

    # Determine the maximum scene size of the image (m)
    data.maxWr = c / (2 * data.deltaF)
    data.maxWx = c / (2 * data.deltaAz * np.mean(data.minF))

    # Determine the resolution of the image (m)
    data.dr = c / (2 * data.deltaF * data.K)
    data.dx = c / (2 * data.totalAz * np.mean(data.minF))

    # Display maximum scene size and resolution
    print(f"Maximum Scene Size: {data.maxWr:.2f} m range, {data.maxWx:.2f} m cross-range")
    print(f"Resolution: {data.dr:.2f} m range, {data.dx:.2f} m cross-range")

    # Calculate the range to every bin in the range profile (m)
    data.r_vec = np.linspace(-data.Nfft/2, data.Nfft/2 - 1, data.Nfft) * data.maxWr / data.Nfft

    # Initialize the image with all zero values
    data.im_final = np.zeros_like(data.x_mat, dtype=np.complex128)

    # Set up a vector to keep execution times for each pulse (sec)
    t = np.zeros(data.Np)

    # Loop through every pulse
    for ii in range(data.Np):
        # Display status of the imaging process
        if ii > 0:
            t_sofar = np.sum(t[:ii])
            t_est = (t_sofar * data.Np / ii - t_sofar) / 60
            print(f"Pulse {ii+1} of {data.Np}, {t_est:.2f} minutes remaining")
        else:
            print(f"Pulse 1 of {data.Np}")
        start = np.nan; import time; start = time.time()  # start timer

        # Form the range profile with zero padding added
        rc = np.fft.fftshift(np.fft.ifft(data.phdata[:, ii], n=data.Nfft))

        # Calculate differential range for each pixel in the image (m)
        dR = np.sqrt((data.AntX[ii] - data.x_mat)**2 +
                     (data.AntY[ii] - data.y_mat)**2 +
                     (data.AntZ[ii] - data.z_mat)**2) - data.R0[ii]

        # Calculate phase correction for image
        phCorr = np.exp(1j * 4 * np.pi * data.minF[ii] / c * dR)

        # Determine which pixels fall within the range swath
        mask = (dR > data.r_vec.min()) & (dR < data.r_vec.max())

        # Update the image using linear interpolation
        from scipy.interpolate import interp1d
        interp_func = interp1d(data.r_vec, rc, kind='linear', bounds_error=False, fill_value=0)
        data.im_final[mask] += interp_func(dR[mask]) * phCorr[mask]

        # Determine the execution time for this pulse
        t[ii] = time.time() - start

    return data

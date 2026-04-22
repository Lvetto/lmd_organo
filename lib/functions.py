import numpy as np
import pandas as pd

EPS = np.finfo(float).eps

def get_average_spectrum(data, rate, window_size, window_step):
    """Compute an average magnitude spectrum over central signal windows.

        The function analyzes the central half of the input signal, applies a
        Gaussian window to each frame, computes the real FFT magnitude for each
        frame, and returns the mean magnitude spectrum.

        Robustness notes
        ----------------
        - If the central segment is shorter than ``window_size``, the segment is
            zero-padded and one FFT frame is still computed.
        - This design avoids empty-frame averages and prevents NaN propagation.

    Parameters
    ----------
    data : np.ndarray
        Input time-domain signal samples.
    rate : float
        Sampling rate in Hz.
    window_size : int
        Number of samples in each analysis window.
    window_step : int
        Hop size in samples between consecutive windows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Frequency bins in Hz and corresponding averaged FFT magnitudes.
    """

    data = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    if window_size <= 0 or window_step <= 0:
        raise ValueError("window_size and window_step must be > 0")

    rms = np.sqrt(np.convolve(data**2, np.ones(1000) / 1000, mode='same'))
    threshold = np.max(rms) * 0.7
    stable_indices = np.where(rms > threshold)[0]

    if stable_indices.size == 0:

        segment = data
    else:
        start, end = stable_indices[0], stable_indices[-1]
        if end <= start:
            segment = data
        else:
            segment = data[start:end]

    N = window_size
    n = np.arange(N)
    window_gauss = np.exp(-0.5 * ((n - N/2) / (window_size/2))**2)
    
    all_magnitudes = []

    for i in range(0, len(segment) - window_size + 1, window_step):

        window = segment[i : i + window_size] # Extract the current window segment
        windowed_signal = window * window_gauss # Apply the Gaussian window to the segment

        fft_res = np.fft.rfft(windowed_signal) # Compute the real FFT of the windowed signal
        all_magnitudes.append(np.abs(fft_res)) # Store the magnitude spectrum for this window
    
    if not all_magnitudes:
        # If no complete frame is available, zero-pad one frame and compute one FFT.
        padded = np.zeros(window_size, dtype=float)
        segment_clip = segment[:window_size]
        padded[:len(segment_clip)] = segment_clip
        all_magnitudes.append(np.abs(np.fft.rfft(padded * window_gauss)))

    avg_magnitude = np.mean(all_magnitudes, axis=0) # Compute the average magnitude spectrum across all windows
    freqs = np.fft.rfftfreq(window_size, 1/rate) # Compute the corresponding frequency bins for the FFT
    
    return freqs, avg_magnitude

def get_f0_from_peaks(peaks_frequencies):
    """Estimate the fundamental frequency from detected spectral peaks.

        The input peak frequencies are sorted, adjacent differences are computed,
        and the median spacing is returned as an f0 estimate.

        Robustness notes
        ----------------
        - If fewer than two valid peaks are available (or no positive spacing
            exists), the function returns ``0.0`` as a finite fallback.

    Parameters
    ----------
    peaks_frequencies : np.ndarray
        Detected peak frequencies in Hz.

    Returns
    -------
    float
        Estimated fundamental frequency in Hz. Returns ``0.0`` when
        estimation is not possible.
    """

    freqs = np.sort(peaks_frequencies)
    if freqs.size < 2:
        return 0.0

    diffs = np.diff(freqs)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0

    f0_mediana = np.median(diffs)

    return float(max(f0_mediana, 0.0))

def extract_harmonics(frequencies, fft_magnitude, f0, num_harmonics):
    """Extract and normalize harmonic amplitudes from an averaged FFT spectrum.

        For each harmonic i (from 1 to num_harmonics), this function searches
        the interval around i*f0 with a tolerance of 5% of f0 and keeps the
        maximum FFT magnitude found in that interval.

        Robustness notes
        ----------------
        - If ``f0`` is non-finite or non-positive, the function returns a zero
            feature vector.
        - If the first harmonic amplitude is invalid or zero, the function returns
            zeros instead of dividing by zero.
        - Normalization uses ``max(first, EPS)`` for numerical safety.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the FFT (Hz).
    fft_magnitude : np.ndarray
        Magnitude spectrum corresponding to frequencies.
    f0 : float
        Fundamental frequency (Hz).
    num_harmonics : int
        Number of harmonics to extract.

    Returns
    -------
    np.ndarray
        Harmonic amplitudes normalized to the first harmonic. The returned
        vector is always finite.
    """

    if num_harmonics <= 0:
        return np.array([], dtype=float) # Return empty array for non-positive num_harmonics

    if not np.isfinite(f0) or f0 <= 0: # If f0 is not valid, return zeros for all harmonics
        return np.zeros(num_harmonics, dtype=float)

    harmonic_amplitudes = []
    tolerance = f0 * 0.05 

    for i in range(1, num_harmonics + 1):
        target_freq = i * f0
        
        lower_bound = target_freq - tolerance
        upper_bound = target_freq + tolerance

        mask = (frequencies >= lower_bound) & (frequencies <= upper_bound) # Find indices of frequencies within the tolerance range
        
        if np.any(mask): # If there are frequencies in the range, take the maximum magnitude; otherwise, use 0 
            amplitude = np.max(fft_magnitude[mask])
        else:
            amplitude = 0
            
        harmonic_amplitudes.append(amplitude)
    
    harmonic_amplitudes = np.array(harmonic_amplitudes, dtype=float)
    first = harmonic_amplitudes[0]
    if not np.isfinite(first) or first <= 0:
        return np.zeros(num_harmonics, dtype=float)

    harmonic_amplitudes_norm = harmonic_amplitudes / max(first, EPS) # Normalize by the first harmonic, using max with EPS to avoid division by zero
        
    return harmonic_amplitudes_norm

def get_adaptive_params(data, rate):
    """Estimate adaptive FFT analysis parameters from a coarse f0 estimate.

    The function performs a quick spectral analysis on the central half of the
    signal to obtain a rough fundamental-frequency estimate. It then derives an
    adaptive window size so that each analysis frame contains about 8 periods
    of the detected pitch.

    Robustness notes
    ----------------
    - The quick FFT size is capped by the available central segment length.
    - If coarse f0 estimation fails (non-positive value), a default window
      length of ``8192`` is used.
    - The adaptive window size is clamped to ``[4096, 32768]`` samples.

    Parameters
    ----------
    data : np.ndarray
        Input time-domain signal samples.
    rate : float
        Sampling rate in Hz.

    Returns
    -------
    tuple[int, int]
        ``(window_size, window_step)`` where ``window_step`` is half of
        ``window_size``.
    """

    if len(data) < 2:
        return 8192, 4096

    start = len(data) // 4
    end = 3 * len(data) // 4
    quick_seg = data[start:end]
    
    n_quick = 8192
    if len(quick_seg) < n_quick:
        n_quick = len(quick_seg)
    if n_quick < 2:
        return 8192, 4096
    
    f_grezza, mag_grezza = get_average_spectrum(data, rate, n_quick, max(1, n_quick // 2))
    if mag_grezza.size > 1:
        f0_grezza = f_grezza[np.argmax(mag_grezza[1:]) + 1]
    else:
        f0_grezza = 0.0

    if f0_grezza > 0:
        n_adattiva = int(8 * (rate / f0_grezza))
    else:
        n_adattiva = 8192 
    
    n_adattiva = max(4096, min(n_adattiva, 32768))
    
    return n_adattiva, n_adattiva // 2

def find_closest_note(freq, df_mapping):
    """Return the musical note name closest to the provided frequency.

    Parameters
    ----------
    freq : float | int | None
        Estimated frequency in Hz.
    df_mapping : pandas.DataFrame
        Mapping table containing at least two columns:
        - ``Frequenza``: reference frequency in Hz
        - ``Nome``: note label to return

    Returns
    -------
    str | None
        Closest note name if ``freq`` is finite and positive, otherwise ``None``.
    """
    if freq is None or not np.isfinite(freq) or freq <= 0:
        return None

    idx = (df_mapping['Frequenza'] - freq).abs().idxmin()
    return df_mapping.loc[idx, 'Nome']
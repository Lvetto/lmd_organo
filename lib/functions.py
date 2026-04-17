import numpy as np

def get_average_spectrum(data, rate, window_size, window_step):

    start = len(data) // 4
    end = 3 * len(data) // 4
    segment = data[start:end]
    N = window_size
    n = np.arange(N)
    window_gauss = np.exp(-0.5 * ((n - N/2) / (window_size/2))**2)
    
    all_magnitudes = []
    
    for i in range(0, len(segment) - window_size, window_step):

        window = segment[i : i + window_size]
        windowed_signal = window * window_gauss
        

        fft_res = np.fft.rfft(windowed_signal)
        all_magnitudes.append(np.abs(fft_res))
    
    avg_magnitude = np.mean(all_magnitudes, axis=0)
    freqs = np.fft.rfftfreq(window_size, 1/rate)
    
    return freqs, avg_magnitude

def get_f0_from_peaks(peaks_frequencies):

    freqs = np.sort(peaks_frequencies)
    diffs = np.diff(freqs)

    f0_mediana = np.median(diffs)

    return f0_mediana

def extract_harmonics(frequencies, fft_magnitude, f0, num_harmonics):

    harmonic_amplitudes = []
    tolerance = f0 * 0.05 

    for i in range(1, num_harmonics + 1):
        target_freq = i * f0
        
        lower_bound = target_freq - tolerance
        upper_bound = target_freq + tolerance

        mask = (frequencies >= lower_bound) & (frequencies <= upper_bound)
        
        if np.any(mask):
            amplitude = np.max(fft_magnitude[mask])
        else:
            amplitude = 0
            
        harmonic_amplitudes.append(amplitude)
    
    harmonic_amplitudes = np.array(harmonic_amplitudes)
    harmonic_amplitudes_norm = harmonic_amplitudes / harmonic_amplitudes[0]
        
    return harmonic_amplitudes_norm
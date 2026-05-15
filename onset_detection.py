import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from lib.readwav import readwav
import ipywidgets as widgets
import json
import emcee
from scipy.special import logsumexp


def secs_to_samples(secs, fs):
    """Convert seconds to samples."""
    return secs * fs

def samples_to_secs(samples, fs):
    """Convert samples to seconds."""
    return samples / fs

def compute_spectral_flux(segment, frame_size, hop_size):
    """Compute the spectral flux of a signal segment."""
    # Compute the magnitude spectrum
    window = np.hanning(frame_size)
    frames = sp.signal.stft(segment, nperseg=frame_size, noverlap=frame_size - hop_size, window=window)[2]
    magnitude_spectrum = np.abs(frames)
    
    # Compute spectral flux
    spectral_flux = np.sqrt(np.sum(np.diff(magnitude_spectrum, axis=1)**2, axis=0))

    inds = np.arange(len(spectral_flux)) * hop_size

    return inds, spectral_flux

def detect_onsets(signal, fs, convolve_window=5, prominence=1000, distance=10, frame_size=1024, hop_size=512):
    """Detect onsets from the signal."""

    inds, spectral_flux = compute_spectral_flux(signal, frame_size, hop_size)

    smooth_flux = np.convolve(spectral_flux, np.ones(convolve_window)/convolve_window, mode='same')
    flux_peaks, _ = sp.signal.find_peaks(smooth_flux, prominence=prominence, distance=distance)

    return inds[flux_peaks]

def segment_signal(signal, fs, onsets, delta=0.1):
    """Segment the signal based on detected onsets."""
    segments = []
    for i in range(len(onsets) - 1):
        start_time = onsets[i] + secs_to_samples(delta, fs)
        end_time = onsets[i + 1] - secs_to_samples(delta, fs)
        segment = signal[int(start_time):int(end_time)]
        segments.append(segment)
    return segments

def compute_avg_spectrum(segment, fs):
    width = (len(segment) / fs) / 10  # Set width to 1/10th of the segment duration
    gaussian_window = sp.signal.windows.gaussian(int(width * fs), std = 0.2 * fs)
    freqs, avg_spectrum = sp.signal.welch(segment, fs = fs, window = gaussian_window, nperseg = len(gaussian_window))
    return freqs, avg_spectrum

def compute_avg_spectrum_nperiods(segment, fs, n_periods=10):

    peaks, _ = sp.signal.find_peaks(segment, height=np.max(segment)*0.5)

    peaks_diff = np.diff(peaks)
    if len(peaks_diff) == 0:
        return None, None  # Not enough peaks to compute periods
    
    period = np.median(peaks_diff)  # Estimate period as median distance between peaks
    segment_length = len(segment)
    n_samples_per_period = int(period)
    n_samples_to_use = n_samples_per_period * n_periods

    window = sp.signal.windows.gaussian(n_samples_to_use, std=0.2 * n_samples_to_use)

    freqs, avg_spectrum = sp.signal.welch(segment, fs=fs, window=window, nperseg=len(window))

    return freqs, avg_spectrum

def cleanup_segments(segments, fs, min_duration_secs=0.1, window_size=2048):
    """
    Cut each segment again, to remove small amplitude tails at the beginning and end of each segment.
    """
    for i in range(len(segments)):
        try:
            segment = segments[i]
            
            segment_peaks, _ = sp.signal.find_peaks(segment, height=np.max(segment)*0.1)

            if len(segment_peaks) == 0:
                segments[i] = None  # Mark for removal
                continue

            start_sample = segment_peaks[0]
            end_sample = segment_peaks[-1]

            segments[i] = segment[start_sample:end_sample]
        except Exception as e:
            print(f"Error cleaning segment {i}: {e}")
            segments[i] = None  # Mark for removal
    
    # Remove segments that are too short
    segments = [seg for seg in segments if seg is not None]
    return segments

def find_peaks(segment_psd, freqs, prominence=5, height=0, distance=10):

    spectrum = np.log10(segment_psd)

    # remove infs and nans
    spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

    norm_spectrum = spectrum / np.max(spectrum) * 100  # Normalize to percentage

    peaks_params = {
        "prominence": prominence,
        "height": height,
        "distance": distance
    }

    peaks, _ = sp.signal.find_peaks(norm_spectrum, **peaks_params)


    return freqs[peaks]




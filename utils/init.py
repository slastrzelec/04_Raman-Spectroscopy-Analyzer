"""
Utility modules for Raman spectroscopy analysis
"""

from .data_loading import load_spectrum, load_multiple_spectra
from .preprocessing import (
    baseline_linear_endpoints,
    normalize_spectrum,
    smooth_spectrum
)
from .peak_detection import detect_peaks, categorize_peaks
from .peak_fitting import fit_peaks_region, calculate_id_ig_ratio
from .visualization import (
    plot_spectrum,
    plot_baseline_correction,
    plot_normalization_comparison,
    plot_peak_detection,
    plot_peak_fitting
)

__all__ = [
    'load_spectrum',
    'load_multiple_spectra',
    'baseline_linear_endpoints',
    'normalize_spectrum',
    'smooth_spectrum',
    'detect_peaks',
    'categorize_peaks',
    'fit_peaks_region',
    'calculate_id_ig_ratio',
    'plot_spectrum',
    'plot_baseline_correction',
    'plot_normalization_comparison',
    'plot_peak_detection',
    'plot_peak_fitting'
]
"""
Functions for peak detection in Raman spectra
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_peaks(df, height=0.15, prominence=0.08, distance=50, n_peaks=None):
    """
    Detect peaks in Raman spectrum
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Wavenumber' and 'Intensity' columns
    height : float
        Minimum peak height
    prominence : float
        Minimum peak prominence
    distance : int
        Minimum distance between peaks (in data points)
    n_peaks : int or None
        Number of top peaks to return (None = all peaks)
    
    Returns:
    --------
    np.ndarray
        Array of peak indices
    pd.DataFrame
        DataFrame with peak information
    dict
        Detection parameters
    """
    # Find all peaks
    peaks, properties = find_peaks(
        df['Intensity'].values,
        height=height,
        prominence=prominence,
        distance=distance
    )
    
    if len(peaks) == 0:
        return np.array([]), pd.DataFrame(), {'n_peaks_found': 0}
    
    # Get peak intensities
    peak_intensities = df.loc[peaks, 'Intensity'].values
    
    # If n_peaks specified, select top n by intensity
    if n_peaks is not None and len(peaks) > n_peaks:
        sorted_indices = np.argsort(peak_intensities)[::-1]  # Descending
        top_peaks = peaks[sorted_indices[:n_peaks]]
        top_peaks = np.sort(top_peaks)  # Sort by wavenumber
    else:
        top_peaks = peaks
    
    # Create peak information DataFrame
    peak_data = []
    for i, peak_idx in enumerate(top_peaks):
        wavenumber = df.loc[peak_idx, 'Wavenumber']
        intensity = df.loc[peak_idx, 'Intensity']
        
        peak_data.append({
            'Peak_Number': i + 1,
            'Index': peak_idx,
            'Wavenumber': wavenumber,
            'Intensity': intensity,
            'Prominence': properties['prominences'][np.where(peaks == peak_idx)[0][0]] if peak_idx in peaks else None
        })
    
    peak_df = pd.DataFrame(peak_data)
    
    params = {
        'height': height,
        'prominence': prominence,
        'distance': distance,
        'n_peaks_requested': n_peaks,
        'n_peaks_found': len(top_peaks),
        'n_peaks_total': len(peaks)
    }
    
    return top_peaks, peak_df, params


def categorize_peaks(peak_df, bands_config=None):
    """
    Categorize peaks by spectral region
    
    Parameters:
    -----------
    peak_df : pd.DataFrame
        DataFrame with peak information
    bands_config : dict
        Dictionary with band definitions
        Default: CNT D, G, 2D bands
    
    Returns:
    --------
    dict
        Peaks categorized by region
    """
    if bands_config is None:
        bands_config = {
            'D-band': {'range': (1200, 1450), 'description': 'Defects/disorder'},
            'G-band': {'range': (1450, 1800), 'description': 'Graphitic carbon'},
            '2D-band': {'range': (2500, 3000), 'description': 'Second-order'},
            'Other': {'range': (0, 10000), 'description': 'Other features'}
        }
    
    categorized = {}
    
    for band_name, band_info in bands_config.items():
        min_wn, max_wn = band_info['range']
        
        band_peaks = peak_df[
            (peak_df['Wavenumber'] >= min_wn) & 
            (peak_df['Wavenumber'] <= max_wn)
        ].copy()
        
        if len(band_peaks) > 0:
            categorized[band_name] = {
                'peaks': band_peaks,
                'count': len(band_peaks),
                'description': band_info['description']
            }
    
    return categorized


def find_closest_peak(peak_df, target_wavenumber, tolerance=50):
    """
    Find peak closest to target wavenumber
    
    Parameters:
    -----------
    peak_df : pd.DataFrame
        DataFrame with peak information
    target_wavenumber : float
        Target wavenumber
    tolerance : float
        Maximum distance from target
    
    Returns:
    --------
    pd.Series or None
        Peak information or None if not found
    """
    if peak_df.empty:
        return None
    
    differences = np.abs(peak_df['Wavenumber'] - target_wavenumber)
    min_diff_idx = differences.idxmin()
    
    if differences[min_diff_idx] <= tolerance:
        return peak_df.loc[min_diff_idx]
    else:
        return None


def get_peak_statistics(peak_df):
    """
    Calculate statistics for detected peaks
    
    Parameters:
    -----------
    peak_df : pd.DataFrame
        DataFrame with peak information
    
    Returns:
    --------
    dict
        Dictionary with statistics
    """
    if peak_df.empty:
        return {}
    
    stats = {
        'n_peaks': len(peak_df),
        'mean_intensity': peak_df['Intensity'].mean(),
        'std_intensity': peak_df['Intensity'].std(),
        'max_intensity': peak_df['Intensity'].max(),
        'min_intensity': peak_df['Intensity'].min(),
        'strongest_peak_wavenumber': peak_df.loc[peak_df['Intensity'].idxmax(), 'Wavenumber'],
        'wavenumber_range': (peak_df['Wavenumber'].min(), peak_df['Wavenumber'].max()),
        'mean_peak_spacing': np.mean(np.diff(peak_df['Wavenumber'].values)) if len(peak_df) > 1 else None
    }
    
    return stats


def filter_peaks_by_region(peak_df, min_wavenumber, max_wavenumber):
    """
    Filter peaks within a wavenumber region
    
    Parameters:
    -----------
    peak_df : pd.DataFrame
        DataFrame with peak information
    min_wavenumber : float
        Minimum wavenumber
    max_wavenumber : float
        Maximum wavenumber
    
    Returns:
    --------
    pd.DataFrame
        Filtered peaks
    """
    return peak_df[
        (peak_df['Wavenumber'] >= min_wavenumber) & 
        (peak_df['Wavenumber'] <= max_wavenumber)
    ].copy()
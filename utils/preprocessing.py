"""
Functions for preprocessing Raman spectra
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_linear_endpoints(df, n_points=5):
    """
    Linear baseline correction using first and last n points
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Wavenumber' and 'Intensity' columns
    n_points : int
        Number of points to use from each end
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with corrected spectrum
    np.ndarray
        Calculated baseline
    dict
        Baseline parameters
    """
    x = df['Wavenumber'].values
    y = df['Intensity'].values
    
    # Take first and last n points
    x_endpoints = np.concatenate([x[:n_points], x[-n_points:]])
    y_endpoints = np.concatenate([y[:n_points], y[-n_points:]])
    
    # Fit linear function: y = ax + b
    coeffs = np.polyfit(x_endpoints, y_endpoints, 1)
    a, b = coeffs
    
    # Calculate baseline for all points
    baseline = a * x + b
    
    # Subtract baseline
    df_corrected = df.copy()
    df_corrected['Intensity'] = y - baseline
    
    params = {
        'method': 'linear_endpoints',
        'n_points': n_points,
        'slope': a,
        'intercept': b
    }
    
    return df_corrected, baseline, params


def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """
    Asymmetric Least Squares Smoothing baseline correction
    
    Parameters:
    -----------
    y : np.ndarray
        Input signal
    lam : float
        Smoothness parameter
    p : float
        Asymmetry parameter
    niter : int
        Number of iterations
    
    Returns:
    --------
    np.ndarray
        Baseline
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    
    return z


def baseline_als_correction(df, lam=1e6, p=0.01, niter=10):
    """
    Apply ALS baseline correction to DataFrame
    """
    x = df['Wavenumber'].values
    y = df['Intensity'].values
    
    baseline = baseline_als(y, lam=lam, p=p, niter=niter)
    
    df_corrected = df.copy()
    df_corrected['Intensity'] = y - baseline
    
    params = {
        'method': 'als',
        'lambda': lam,
        'p': p,
        'iterations': niter
    }
    
    return df_corrected, baseline, params


def normalize_spectrum(df, method='minmax'):
    """
    Normalize Raman spectrum
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with spectrum
    method : str
        Normalization method ('minmax', 'max', 'area', 'vector')
    
    Returns:
    --------
    pd.DataFrame
        Normalized spectrum
    dict
        Normalization parameters
    """
    df_normalized = df.copy()
    intensity = df['Intensity'].values
    
    if method == 'minmax':
        min_val = intensity.min()
        max_val = intensity.max()
        normalized = (intensity - min_val) / (max_val - min_val)
        params = {'method': 'minmax', 'min': min_val, 'max': max_val}
        
    elif method == 'max':
        max_val = intensity.max()
        normalized = intensity / max_val
        params = {'method': 'max', 'max': max_val}
        
    elif method == 'area':
        from scipy.integrate import trapezoid
        area = trapezoid(intensity, df['Wavenumber'].values)
        normalized = intensity / area
        params = {'method': 'area', 'area': area}
        
    elif method == 'vector':
        norm = np.linalg.norm(intensity)
        normalized = intensity / norm
        params = {'method': 'vector', 'norm': norm}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    df_normalized['Intensity'] = normalized
    
    return df_normalized, params


def smooth_spectrum(df, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay smoothing filter
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with spectrum
    window_length : int
        Window length (must be odd)
    polyorder : int
        Polynomial order
    
    Returns:
    --------
    pd.DataFrame
        Smoothed spectrum
    dict
        Smoothing parameters
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    df_smooth = df.copy()
    df_smooth['Intensity'] = savgol_filter(
        df['Intensity'].values,
        window_length=window_length,
        polyorder=polyorder
    )
    
    params = {
        'method': 'savgol',
        'window_length': window_length,
        'polyorder': polyorder
    }
    
    return df_smooth, params
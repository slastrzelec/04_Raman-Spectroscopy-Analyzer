"""
Functions for peak fitting and deconvolution
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def lorentzian(x, amplitude, center, width):
    """
    Lorentzian (Cauchy) function
    
    Parameters:
    -----------
    x : np.ndarray
        X values
    amplitude : float
        Peak amplitude
    center : float
        Peak center
    width : float
        Peak width parameter
    
    Returns:
    --------
    np.ndarray
        Y values
    """
    return amplitude / (1 + ((x - center) / width)**2)


def gaussian(x, amplitude, center, width):
    """
    Gaussian function
    
    Parameters:
    -----------
    x : np.ndarray
        X values
    amplitude : float
        Peak amplitude
    center : float
        Peak center
    width : float
        Standard deviation
    
    Returns:
    --------
    np.ndarray
        Y values
    """
    return amplitude * np.exp(-((x - center)**2) / (2 * width**2))


def multi_peak_function(x, *params, peak_type='lorentzian'):
    """
    Sum of multiple peaks
    
    Parameters:
    -----------
    x : np.ndarray
        X values
    params : tuple
        Flattened parameters [amp1, center1, width1, amp2, center2, width2, ...]
    peak_type : str
        'lorentzian' or 'gaussian'
    
    Returns:
    --------
    np.ndarray
        Sum of all peaks
    """
    n_peaks = len(params) // 3
    y = np.zeros_like(x, dtype=float)
    
    for i in range(n_peaks):
        amp = params[i*3]
        center = params[i*3 + 1]
        width = params[i*3 + 2]
        
        if peak_type == 'lorentzian':
            y += lorentzian(x, amp, center, width)
        elif peak_type == 'gaussian':
            y += gaussian(x, amp, center, width)
    
    return y


def fit_peaks_region(df, region_range, n_peaks, peak_type='lorentzian', 
                     initial_centers=None, max_iterations=10000):
    """
    Fit multiple peaks in a specific region
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Wavenumber' and 'Intensity' columns
    region_range : tuple
        (min_wavenumber, max_wavenumber)
    n_peaks : int
        Number of peaks to fit
    peak_type : str
        'lorentzian' or 'gaussian'
    initial_centers : list or None
        Initial guess for peak centers
    max_iterations : int
        Maximum number of fitting iterations
    
    Returns:
    --------
    dict
        Dictionary with fitting results
    """
    # Extract region
    mask = (df['Wavenumber'] >= region_range[0]) & (df['Wavenumber'] <= region_range[1])
    x_data = df.loc[mask, 'Wavenumber'].values
    y_data = df.loc[mask, 'Intensity'].values
    
    if len(x_data) < 10:
        return {
            'success': False,
            'error': 'Insufficient data points in region'
        }
    
    # Find peaks for initial guess if not provided
    if initial_centers is None:
        peaks, _ = find_peaks(y_data, prominence=0.05)
        if len(peaks) >= n_peaks:
            peak_heights = y_data[peaks]
            top_peak_indices = peaks[np.argsort(peak_heights)[-n_peaks:]]
            initial_centers = sorted(x_data[top_peak_indices])
        else:
            # Distribute evenly
            initial_centers = np.linspace(region_range[0], region_range[1], n_peaks)
    
    # Initial parameters: [amplitude, center, width] for each peak
    initial_params = []
    for center in initial_centers:
        initial_params.extend([
            0.5,   # amplitude
            center,  # center
            20.0   # width
        ])
    
    # Bounds for parameters
    bounds_lower = []
    bounds_upper = []
    for i in range(n_peaks):
        bounds_lower.extend([0.0, region_range[0], 1.0])
        bounds_upper.extend([2.0, region_range[1], 100.0])
    
    try:
        # Fit the peaks
        fitted_params, covariance = curve_fit(
            lambda x, *params: multi_peak_function(x, *params, peak_type=peak_type),
            x_data, y_data,
            p0=initial_params,
            bounds=(bounds_lower, bounds_upper),
            maxfev=max_iterations
        )
        
        # Calculate fitted curve
        y_fitted = multi_peak_function(x_data, *fitted_params, peak_type=peak_type)
        
        # Calculate residuals and R²
        residuals = y_data - y_fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Extract individual peak parameters
        peak_info = []
        individual_peaks = []
        
        for i in range(n_peaks):
            amp = fitted_params[i*3]
            center = fitted_params[i*3 + 1]
            width = fitted_params[i*3 + 2]
            
            # Calculate FWHM
            if peak_type == 'lorentzian':
                fwhm = 2 * width
                area = np.pi * amp * width
            else:  # gaussian
                fwhm = 2.355 * width
                area = amp * width * np.sqrt(2 * np.pi)
            
            # Calculate individual peak curve
            if peak_type == 'lorentzian':
                y_individual = lorentzian(x_data, amp, center, width)
            else:
                y_individual = gaussian(x_data, amp, center, width)
            
            individual_peaks.append({
                'x': x_data,
                'y': y_individual
            })
            
            peak_info.append({
                'Peak': i + 1,
                'Center': center,
                'Amplitude': amp,
                'Width': width,
                'FWHM': fwhm,
                'Area': area
            })
        
        peak_df = pd.DataFrame(peak_info)
        
        # Calculate parameter uncertainties
        try:
            perr = np.sqrt(np.diag(covariance))
            uncertainties = []
            for i in range(n_peaks):
                uncertainties.append({
                    'Peak': i + 1,
                    'Amplitude_err': perr[i*3],
                    'Center_err': perr[i*3 + 1],
                    'Width_err': perr[i*3 + 2]
                })
            uncertainty_df = pd.DataFrame(uncertainties)
        except:
            uncertainty_df = None
        
        return {
            'success': True,
            'fitted_params': fitted_params,
            'x_data': x_data,
            'y_data': y_data,
            'y_fitted': y_fitted,
            'residuals': residuals,
            'individual_peaks': individual_peaks,
            'peak_info': peak_df,
            'uncertainties': uncertainty_df,
            'r_squared': r_squared,
            'rmse': rmse,
            'peak_type': peak_type,
            'region_range': region_range,
            'n_peaks': n_peaks
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def calculate_id_ig_ratio(d_fit_result, g_fit_result, method='area'):
    """
    Calculate ID/IG ratio from fitted D and G bands
    
    Parameters:
    -----------
    d_fit_result : dict
        Fitting result for D-band region
    g_fit_result : dict
        Fitting result for G-band region
    method : str
        'area' or 'height' for ratio calculation
    
    Returns:
    --------
    dict
        Dictionary with ID/IG analysis results
    """
    if not d_fit_result['success'] or not g_fit_result['success']:
        return {
            'success': False,
            'error': 'One or both fits failed'
        }
    
    d_peaks = d_fit_result['peak_info']
    g_peaks = g_fit_result['peak_info']
    
    if d_peaks.empty or g_peaks.empty:
        return {
            'success': False,
            'error': 'No peaks found in one or both regions'
        }
    
    # Find main D and G peaks (highest amplitude/area)
    if method == 'area':
        d_peak = d_peaks.loc[d_peaks['Area'].idxmax()]
        g_peak = g_peaks.loc[g_peaks['Area'].idxmax()]
        ratio = d_peak['Area'] / g_peak['Area']
    else:  # height
        d_peak = d_peaks.loc[d_peaks['Amplitude'].idxmax()]
        g_peak = g_peaks.loc[g_peaks['Amplitude'].idxmax()]
        ratio = d_peak['Amplitude'] / g_peak['Amplitude']
    
    # Estimate crystallite size (La) using Tuinstra-Koenig relation
    # La (nm) ≈ C_λ / (ID/IG) where C_λ depends on laser wavelength
    # For 514 nm: C_λ ≈ 4.4 nm
    # For 532 nm: C_λ ≈ 4.4 nm
    # For 633 nm: C_λ ≈ 2.4 nm
    # For 785 nm: C_λ ≈ 1.8 nm
    C_lambda = 4.4  # Assuming 514-532 nm laser
    la = C_lambda / ratio if ratio > 0 else None
    
    # Material quality assessment
    if ratio < 0.8:
        quality = "High quality (low defects)"
    elif ratio < 1.2:
        quality = "Good quality (moderate defects)"
    else:
        quality = "Lower quality or highly functionalized (high defects)"
    
    return {
        'success': True,
        'ratio': ratio,
        'method': method,
        'd_band': {
            'center': d_peak['Center'],
            'amplitude': d_peak['Amplitude'],
            'fwhm': d_peak['FWHM'],
            'area': d_peak['Area']
        },
        'g_band': {
            'center': g_peak['Center'],
            'amplitude': g_peak['Amplitude'],
            'fwhm': g_peak['FWHM'],
            'area': g_peak['Area']
        },
        'crystallite_size_nm': la,
        'quality_assessment': quality
    }


def calculate_i2d_ig_ratio(fit_2d, fit_g):
    """
    Calculate I2D/IG ratio (useful for graphene layer estimation)
    
    Parameters:
    -----------
    fit_2d : dict
        Fitting result for 2D-band region
    fit_g : dict
        Fitting result for G-band region
    
    Returns:
    --------
    dict
        Dictionary with I2D/IG analysis
    """
    if not fit_2d['success'] or not fit_g['success']:
        return {
            'success': False,
            'error': 'One or both fits failed'
        }
    
    peaks_2d = fit_2d['peak_info']
    peaks_g = fit_g['peak_info']
    
    if peaks_2d.empty or peaks_g.empty:
        return {
            'success': False,
            'error': 'No peaks found'
        }
    
    # Find main peaks
    peak_2d = peaks_2d.loc[peaks_2d['Amplitude'].idxmax()]
    peak_g = peaks_g.loc[peaks_g['Amplitude'].idxmax()]
    
    ratio_height = peak_2d['Amplitude'] / peak_g['Amplitude']
    ratio_area = peak_2d['Area'] / peak_g['Area']
    
    # Estimate number of graphene layers
    # Rough guide: I2D/IG > 2: monolayer, 1-2: bilayer, <1: few-layer
    if ratio_area > 2:
        layer_estimate = "Monolayer graphene"
    elif ratio_area > 1:
        layer_estimate = "Bilayer graphene"
    elif ratio_area > 0.5:
        layer_estimate = "Few-layer graphene (3-5 layers)"
    else:
        layer_estimate = "Multilayer graphene/CNT (>5 layers)"
    
    return {
        'success': True,
        'ratio_height': ratio_height,
        'ratio_area': ratio_area,
        '2d_band': {
            'center': peak_2d['Center'],
            'amplitude': peak_2d['Amplitude'],
            'fwhm': peak_2d['FWHM']
        },
        'g_band': {
            'center': peak_g['Center'],
            'amplitude': peak_g['Amplitude'],
            'fwhm': peak_g['FWHM']
        },
        'layer_estimate': layer_estimate
    }
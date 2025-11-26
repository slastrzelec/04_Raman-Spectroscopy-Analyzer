"""
Functions for loading Raman spectroscopy data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st


def load_spectrum(file_path, sep=r'\s+', column_names=None):
    """
    Load a single Raman spectrum from a text file
    
    Parameters:
    -----------
    file_path : str or Path or UploadedFile
        Path to the spectrum file or Streamlit UploadedFile object
    sep : str
        Separator for the data (default: whitespace)
    column_names : list
        Column names (default: ['Wavenumber', 'Intensity'])
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with spectrum data
    dict
        Metadata about the spectrum
    """
    if column_names is None:
        column_names = ['Wavenumber', 'Intensity']
    
    try:
        # Handle Streamlit UploadedFile
        if hasattr(file_path, 'read'):
            df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)
            filename = file_path.name
        else:
            df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)
            filename = Path(file_path).name
        
        # Create metadata
        metadata = {
            'filename': filename,
            'n_points': len(df),
            'wavenumber_range': (df['Wavenumber'].min(), df['Wavenumber'].max()),
            'intensity_range': (df['Intensity'].min(), df['Intensity'].max()),
            'wavenumber_step': np.mean(np.diff(df['Wavenumber'].values))
        }
        
        return df, metadata
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None


def load_multiple_spectra(file_list, sep=r'\s+'):
    """
    Load multiple Raman spectra
    
    Parameters:
    -----------
    file_list : list
        List of file paths or UploadedFile objects
    sep : str
        Separator for the data
    
    Returns:
    --------
    dict
        Dictionary with filename as key and (DataFrame, metadata) as value
    """
    spectra = {}
    
    for file in file_list:
        df, metadata = load_spectrum(file, sep=sep)
        if df is not None:
            if hasattr(file, 'name'):
                key = file.name
            else:
                key = Path(file).name
            spectra[key] = {'data': df, 'metadata': metadata}
    
    return spectra


def validate_spectrum(df):
    """
    Validate spectrum data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Spectrum data
    
    Returns:
    --------
    bool
        True if valid
    list
        List of warnings/errors
    """
    issues = []
    
    if df is None or df.empty:
        issues.append("ERROR: DataFrame is empty")
        return False, issues
    
    if 'Wavenumber' not in df.columns or 'Intensity' not in df.columns:
        issues.append("ERROR: Required columns missing")
        return False, issues
    
    if df['Wavenumber'].isnull().any():
        issues.append("WARNING: Missing wavenumber values detected")
    
    if df['Intensity'].isnull().any():
        issues.append("WARNING: Missing intensity values detected")
    
    if len(df) < 100:
        issues.append("WARNING: Very few data points (< 100)")
    
    if not df['Wavenumber'].is_monotonic_increasing:
        issues.append("WARNING: Wavenumber values are not monotonically increasing")
    
    return len([i for i in issues if i.startswith("ERROR")]) == 0, issues


def export_spectrum(df, filename, format='csv'):
    """
    Export spectrum data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Spectrum data
    filename : str
        Output filename
    format : str
        Export format ('csv', 'excel', 'json')
    
    Returns:
    --------
    bytes
        File content as bytes for download
    """
    if format.lower() == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format.lower() == 'excel':
        from io import BytesIO
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        return output.getvalue()
    elif format.lower() == 'json':
        return df.to_json(orient='records', indent=2).encode('utf-8')
    else:
        raise ValueError(f"Unsupported format: {format}")
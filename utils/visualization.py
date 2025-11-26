"""
Visualization functions for Raman spectroscopy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


def plot_spectrum(df, title="Raman Spectrum", interactive=True, 
                  show_grid=True, height=500):
    """
    Plot a single Raman spectrum
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Wavenumber' and 'Intensity' columns
    title : str
        Plot title
    interactive : bool
        If True, use Plotly (interactive), else Matplotlib
    show_grid : bool
        Show grid lines
    height : int
        Plot height in pixels (for Plotly)
    
    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if interactive:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Wavenumber'],
            y=df['Intensity'],
            mode='lines',
            name='Spectrum',
            line=dict(color='blue', width=1.5)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Raman shift (cm⁻¹)',
            yaxis_title='Intensity (a.u.)',
            height=height,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        if show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(df['Wavenumber'], df['Intensity'], 'b-', linewidth=1.5)
        ax.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def plot_baseline_correction(df_original, df_corrected, baseline, 
                             n_points=None, interactive=True):
    """
    Plot original spectrum with baseline and corrected spectrum
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original spectrum
    df_corrected : pd.DataFrame
        Baseline-corrected spectrum
    baseline : np.ndarray
        Calculated baseline
    n_points : int or None
        Number of endpoint points used (for annotation)
    interactive : bool
        Use Plotly or Matplotlib
    
    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if interactive:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original Spectrum with Baseline', 
                          'Baseline-Corrected Spectrum'),
            vertical_spacing=0.12
        )
        
        # Original with baseline
        fig.add_trace(
            go.Scatter(x=df_original['Wavenumber'], y=df_original['Intensity'],
                      mode='lines', name='Original', line=dict(color='blue', width=1.5)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_original['Wavenumber'], y=baseline,
                      mode='lines', name='Baseline', 
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Add endpoint markers if n_points provided
        if n_points is not None:
            fig.add_trace(
                go.Scatter(x=df_original['Wavenumber'].iloc[:n_points],
                          y=df_original['Intensity'].iloc[:n_points],
                          mode='markers', name=f'First {n_points} points',
                          marker=dict(color='green', size=8)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_original['Wavenumber'].iloc[-n_points:],
                          y=df_original['Intensity'].iloc[-n_points:],
                          mode='markers', name=f'Last {n_points} points',
                          marker=dict(color='magenta', size=8)),
                row=1, col=1
            )
        
        # Corrected spectrum
        fig.add_trace(
            go.Scatter(x=df_corrected['Wavenumber'], y=df_corrected['Intensity'],
                      mode='lines', name='Corrected', line=dict(color='green', width=1.5)),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=2, col=1)
        
        fig.update_xaxes(title_text="Raman shift (cm⁻¹)", row=2, col=1)
        fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
        fig.update_yaxes(title_text="Intensity (a.u.)", row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Original with baseline
        ax1.plot(df_original['Wavenumber'], df_original['Intensity'], 
                'b-', linewidth=1.5, label='Original', alpha=0.7)
        ax1.plot(df_original['Wavenumber'], baseline, 
                'r--', linewidth=2, label='Baseline')
        
        if n_points is not None:
            ax1.plot(df_original['Wavenumber'].iloc[:n_points],
                    df_original['Intensity'].iloc[:n_points],
                    'go', markersize=8, label=f'First {n_points} points')
            ax1.plot(df_original['Wavenumber'].iloc[-n_points:],
                    df_original['Intensity'].iloc[-n_points:],
                    'mo', markersize=8, label=f'Last {n_points} points')
        
        ax1.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax1.set_title('Original Spectrum with Baseline', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Corrected
        ax2.plot(df_corrected['Wavenumber'], df_corrected['Intensity'], 
                'g-', linewidth=1.5, label='Corrected')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
        ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax2.set_title('Baseline-Corrected Spectrum', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def plot_normalization_comparison(df_original, df_normalized, method, interactive=True):
    """
    Compare original and normalized spectra
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original spectrum
    df_normalized : pd.DataFrame
        Normalized spectrum
    method : str
        Normalization method used
    interactive : bool
        Use Plotly or Matplotlib
    
    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if interactive:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Spectrum', 
                          f'Normalized Spectrum ({method})'),
            horizontal_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df_original['Wavenumber'], y=df_original['Intensity'],
                      mode='lines', name='Original', line=dict(color='blue', width=1.5)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_normalized['Wavenumber'], y=df_normalized['Intensity'],
                      mode='lines', name='Normalized', line=dict(color='green', width=1.5)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Raman shift (cm⁻¹)", row=1, col=1)
        fig.update_xaxes(title_text="Raman shift (cm⁻¹)", row=1, col=2)
        fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
        fig.update_yaxes(title_text="Normalized Intensity", row=1, col=2)
        
        fig.update_layout(
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.plot(df_original['Wavenumber'], df_original['Intensity'], 
                'b-', linewidth=1.5)
        ax1.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax1.set_title('Original Spectrum', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(df_normalized['Wavenumber'], df_normalized['Intensity'], 
                'g-', linewidth=1.5)
        ax2.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
        ax2.set_ylabel('Normalized Intensity', fontsize=12)
        ax2.set_title(f'Normalized Spectrum ({method})', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def plot_peak_detection(df, peaks, peak_df, interactive=True):
    """
    Plot spectrum with detected peaks
    
    Parameters:
    -----------
    df : pd.DataFrame
        Spectrum data
    peaks : np.ndarray
        Array of peak indices
    peak_df : pd.DataFrame
        DataFrame with peak information
    interactive : bool
        Use Plotly or Matplotlib
    
    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if interactive:
        fig = go.Figure()
        
        # Spectrum
        fig.add_trace(go.Scatter(
            x=df['Wavenumber'],
            y=df['Intensity'],
            mode='lines',
            name='Spectrum',
            line=dict(color='blue', width=1.5)
        ))
        
        # Peaks
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=df.loc[peaks, 'Wavenumber'],
                y=df.loc[peaks, 'Intensity'],
                mode='markers',
                name='Detected Peaks',
                marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
            ))
            
            # Add annotations for peaks
            for _, peak in peak_df.iterrows():
                fig.add_annotation(
                    x=peak['Wavenumber'],
                    y=peak['Intensity'],
                    text=f"{peak['Wavenumber']:.0f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="red",
                    ax=0,
                    ay=-30,
                    bgcolor="yellow",
                    opacity=0.7
                )
        
        fig.update_layout(
            title=f'Peak Detection ({len(peaks)} peaks found)',
            xaxis_title='Raman shift (cm⁻¹)',
            yaxis_title='Intensity (a.u.)',
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(df['Wavenumber'], df['Intensity'], 'b-', linewidth=1.5, label='Spectrum')
        
        if len(peaks) > 0:
            ax.plot(df.loc[peaks, 'Wavenumber'], df.loc[peaks, 'Intensity'], 
                   'rx', markersize=12, markeredgewidth=2, label='Detected Peaks')
            
            # Annotate peaks
            for _, peak in peak_df.iterrows():
                ax.annotate(f"{peak['Wavenumber']:.0f}",
                          xy=(peak['Wavenumber'], peak['Intensity']),
                          xytext=(5, 5),
                          textcoords='offset points',
                          fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.4', 
                                  facecolor='yellow', alpha=0.7),
                          arrowprops=dict(arrowstyle='->', color='black', lw=1))
        
        ax.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title(f'Peak Detection ({len(peaks)} peaks found)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def plot_peak_fitting(fit_result, interactive=True):
    """
    Plot peak fitting results with deconvolution
    
    Parameters:
    -----------
    fit_result : dict
        Dictionary with fitting results from fit_peaks_region
    interactive : bool
        Use Plotly or Matplotlib
    
    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if not fit_result['success']:
        st.error(f"Fitting failed: {fit_result.get('error', 'Unknown error')}")
        return None
    
    x_data = fit_result['x_data']
    y_data = fit_result['y_data']
    y_fitted = fit_result['y_fitted']
    residuals = fit_result['residuals']
    individual_peaks = fit_result['individual_peaks']
    r_squared = fit_result['r_squared']
    peak_type = fit_result['peak_type']
    region_range = fit_result['region_range']
    
    if interactive:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'Peak Deconvolution ({peak_type.capitalize()}) - '
                f'{region_range[0]:.0f}-{region_range[1]:.0f} cm⁻¹ (R² = {r_squared:.4f})',
                'Residuals'
            ),
            vertical_spacing=0.15,
            row_heights=[0.75, 0.25]
        )
        
        # Experimental data
        fig.add_trace(
            go.Scatter(x=x_data, y=y_data,
                      mode='lines', name='Experimental',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Fitted curve
        fig.add_trace(
            go.Scatter(x=x_data, y=y_fitted,
                      mode='lines', name='Fitted',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Individual peaks
        colors = px.colors.qualitative.Set3
        for i, peak_data in enumerate(individual_peaks):
            fig.add_trace(
                go.Scatter(x=peak_data['x'], y=peak_data['y'],
                          mode='lines', name=f'Peak {i+1}',
                          line=dict(color=colors[i % len(colors)], 
                                  width=1.5, dash='dash')),
                row=1, col=1
            )
        
        # Residuals
        fig.add_trace(
            go.Scatter(x=x_data, y=residuals,
                      mode='lines', name='Residuals',
                      line=dict(color='green', width=1.5)),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", 
                     opacity=0.3, row=2, col=1)
        
        fig.update_xaxes(title_text="Raman shift (cm⁻¹)", row=2, col=1)
        fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        ax1.plot(x_data, y_data, 'b-', linewidth=2, label='Experimental', alpha=0.7)
        ax1.plot(x_data, y_fitted, 'r-', linewidth=2, 
                label=f'Fitted (R² = {r_squared:.4f})')
        
        # Individual peaks
        colors = plt.cm.Set3(np.linspace(0, 1, len(individual_peaks)))
        for i, peak_data in enumerate(individual_peaks):
            center = fit_result['peak_info'].iloc[i]['Center']
            ax1.plot(peak_data['x'], peak_data['y'], '--', 
                    color=colors[i], linewidth=1.5, 
                    label=f'Peak {i+1}: {center:.1f} cm⁻¹', alpha=0.8)
        
        ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax1.set_title(f'Peak Deconvolution ({peak_type.capitalize()}) - '
                     f'{region_range[0]:.0f}-{region_range[1]:.0f} cm⁻¹', 
                     fontsize=14)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        ax2.plot(x_data, residuals, 'g-', linewidth=1.5, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
        ax2.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def plot_multiple_spectra(spectra_dict, title="Multiple Spectra Comparison", 
                          interactive=True, normalize=False):
    """
    Plot multiple spectra on the same plot
    
    Parameters:
    -----------
    spectra_dict : dict
        Dictionary with {name: df} pairs
    title : str
        Plot title
    interactive : bool
        Use Plotly or Matplotlib
    normalize : bool
        Normalize all spectra for comparison
    
    Returns:
    --------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if interactive:
        fig = go.Figure()
        
        colors = px.colors.qualitative.Plotly
        
        for i, (name, df) in enumerate(spectra_dict.items()):
            intensity = df['Intensity'].values
            if normalize:
                intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            
            fig.add_trace(go.Scatter(
                x=df['Wavenumber'],
                y=intensity,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Raman shift (cm⁻¹)',
            yaxis_title='Normalized Intensity' if normalize else 'Intensity (a.u.)',
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for name, df in spectra_dict.items():
            intensity = df['Intensity'].values
            if normalize:
                intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            
            ax.plot(df['Wavenumber'], intensity, linewidth=1.5, label=name)
        
        ax.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Normalized Intensity' if normalize else 'Intensity (a.u.)', 
                     fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
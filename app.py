"""
Raman Spectroscopy Analysis Tool
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

# Import utility functions
from utils.data_loading import (
    load_spectrum, load_multiple_spectra, 
    validate_spectrum, export_spectrum
)
from utils.preprocessing import (
    baseline_linear_endpoints, baseline_als_correction,
    normalize_spectrum, smooth_spectrum
)
from utils.peak_detection import (
    detect_peaks, categorize_peaks, 
    get_peak_statistics, filter_peaks_by_region
)
from utils.peak_fitting import (
    fit_peaks_region, calculate_id_ig_ratio, 
    calculate_i2d_ig_ratio
)
from utils.visualization import (
    plot_spectrum, plot_baseline_correction,
    plot_normalization_comparison, plot_peak_detection,
    plot_peak_fitting, plot_multiple_spectra
)
from config.settings import (
    BASELINE_DEFAULTS, NORMALIZATION_DEFAULT,
    PEAK_DETECTION_DEFAULTS, PEAK_FITTING_DEFAULTS,
    CNT_BANDS
)

# Page configuration
st.set_page_config(
    page_title="Raman Spectroscopy Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom CSS - theme is handled by config.toml
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        color: #262730;
    }
    .info-box h3, .info-box h4, .info-box p, .info-box ul, .info-box li {
        color: #262730 !important;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .success-box h3, .success-box h4, .success-box p, .success-box ul, .success-box li {
        color: #155724 !important;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    .warning-box h3, .warning-box h4, .warning-box p, .warning-box ul, .warning-box li {
        color: #856404 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = []

# Main header
st.markdown('<h1 class="main-header">📊 Raman Spectroscopy Analyzer</h1>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔬 Raman Spectroscopy Analyzer")
    
    st.markdown("### 🔬 Navigation")
    
    # File upload section
    st.markdown("#### 📁 Data Upload")
    upload_mode = st.radio(
        "Upload Mode:",
        ["Single File", "Multiple Files"],
        help="Choose whether to analyze a single spectrum or compare multiple spectra"
    )
    
    uploaded_files = st.file_uploader(
        "Upload Raman spectrum file(s) (.txt)",
        type=['txt'],
        accept_multiple_files=(upload_mode == "Multiple Files"),
        help="Upload text files with wavenumber and intensity columns"
    )
    
    # Process uploaded files
    if uploaded_files:
        if upload_mode == "Single File":
            if not isinstance(uploaded_files, list):
                uploaded_files = [uploaded_files]
        
        # Load files
        new_files = {}
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                df, metadata = load_spectrum(file)
                if df is not None:
                    is_valid, issues = validate_spectrum(df)
                    new_files[file.name] = {
                        'raw_data': df,
                        'metadata': metadata,
                        'valid': is_valid,
                        'issues': issues
                    }
        
        st.session_state.uploaded_files.update(new_files)
        
        if len(st.session_state.uploaded_files) > 0 and st.session_state.current_file is None:
            st.session_state.current_file = list(st.session_state.uploaded_files.keys())[0]
    
    # File selector
    if len(st.session_state.uploaded_files) > 0:
        st.markdown("#### 📄 Select File")
        st.session_state.current_file = st.selectbox(
            "Current file:",
            options=list(st.session_state.uploaded_files.keys()),
            index=list(st.session_state.uploaded_files.keys()).index(
                st.session_state.current_file
            ) if st.session_state.current_file else 0
        )
    
    st.markdown("---")
    
    # Settings
    st.markdown("#### ⚙️ Settings")
    interactive_plots = st.checkbox("Interactive Plots (Plotly)", value=True, 
                                    help="Use interactive Plotly charts vs static Matplotlib")
    show_grid = st.checkbox("Show Grid", value=True)
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Raman Spectroscopy Analyzer**
        
        A comprehensive tool for analyzing Raman spectroscopy data:
        - Baseline correction
        - Normalization
        - Peak detection
        - Peak fitting & deconvolution
        - Material characterization
        
        Version 1.0
        """)

# Main content area
if len(st.session_state.uploaded_files) == 0:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
    <h3>👋 Welcome to Raman Spectroscopy Analyzer!</h3>
    <p>This tool helps you analyze Raman spectroscopy data with advanced processing capabilities.</p>
    
    <h4>Getting Started:</h4>
    <ol>
        <li>Upload your Raman spectrum file(s) using the sidebar</li>
        <li>Choose your processing workflow</li>
        <li>Analyze peaks and extract material properties</li>
        <li>Export results and generate reports</li>
    </ol>
    
    <h4>Supported File Format:</h4>
    <ul>
        <li>Text files (.txt) with two columns: Wavenumber and Intensity</li>
        <li>Whitespace or tab-separated values</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example
    st.markdown("### 📝 Example Data Format")
    example_data = pd.DataFrame({
        'Wavenumber': [603.157, 603.648, 604.139, 604.629, 605.120],
        'Intensity': [32.0, 6.0, 12.0, 15.0, 15.0]
    })
    st.dataframe(example_data, use_container_width=True)

else:
    # Get current file data
    current_data = st.session_state.uploaded_files[st.session_state.current_file]
    df_raw = current_data['raw_data']
    metadata = current_data['metadata']
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🔧 Preprocessing", 
        "🔍 Peak Detection",
        "📈 Peak Fitting",
        "📋 Analysis & Results",
        "💾 Export"
    ])
    
    # TAB 1: Data Overview
    with tab1:
        st.markdown('<h2 class="subheader">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### File Information")
            info_df = pd.DataFrame({
                'Property': ['Filename', 'Number of Points', 'Wavenumber Range', 
                           'Intensity Range', 'Average Step Size'],
                'Value': [
                    metadata['filename'],
                    f"{metadata['n_points']} points",
                    f"{metadata['wavenumber_range'][0]:.1f} - {metadata['wavenumber_range'][1]:.1f} cm⁻¹",
                    f"{metadata['intensity_range'][0]:.2f} - {metadata['intensity_range'][1]:.2f}",
                    f"{metadata['wavenumber_step']:.3f} cm⁻¹"
                ]
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
            
            # Validation status
            if current_data['valid']:
                st.markdown('<div class="success-box">✅ Data validation passed</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Data validation warnings:</div>', 
                          unsafe_allow_html=True)
                for issue in current_data['issues']:
                    st.warning(issue)
        
        with col2:
            st.markdown("#### Quick Stats")
            st.metric("Total Points", metadata['n_points'])
            st.metric("Wavenumber Span", 
                     f"{metadata['wavenumber_range'][1] - metadata['wavenumber_range'][0]:.0f} cm⁻¹")
            st.metric("Max Intensity", f"{metadata['intensity_range'][1]:.2f}")
        
        st.markdown("---")
        
        # Plot raw spectrum
        st.markdown("#### Raw Spectrum")
        fig = plot_spectrum(df_raw, title=f"Raw Spectrum - {metadata['filename']}", 
                          interactive=interactive_plots, show_grid=show_grid)
        
        if interactive_plots:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig)
        
        # Data table
        with st.expander("📋 View Raw Data Table"):
            st.dataframe(df_raw, use_container_width=True)
    
    # TAB 2: Preprocessing
    with tab2:
        st.markdown('<h2 class="subheader">🔧 Preprocessing</h2>', unsafe_allow_html=True)
        
        # Initialize processed data if not exists
        if st.session_state.current_file not in st.session_state.processed_data:
            st.session_state.processed_data[st.session_state.current_file] = {
                'baseline_corrected': None,
                'normalized': None,
                'smoothed': None,
                'baseline': None,
                'params': {}
            }
        
        # Baseline Correction Section
        st.markdown("### 1️⃣ Baseline Correction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            baseline_method = st.selectbox(
                "Baseline Correction Method:",
                ["Linear (Endpoints)", "ALS (Asymmetric Least Squares)"],
                help="Choose the baseline correction algorithm"
            )
            
            if baseline_method == "Linear (Endpoints)":
                n_points = st.slider(
                    "Number of endpoint points:",
                    min_value=3,
                    max_value=20,
                    value=BASELINE_DEFAULTS['linear_n_points'],
                    help="Number of points from each end to fit the baseline"
                )
                
                if st.button("Apply Baseline Correction", key="baseline_btn"):
                    df_corrected, baseline, params = baseline_linear_endpoints(df_raw, n_points=n_points)
                    st.session_state.processed_data[st.session_state.current_file]['baseline_corrected'] = df_corrected
                    st.session_state.processed_data[st.session_state.current_file]['baseline'] = baseline
                    st.session_state.processed_data[st.session_state.current_file]['params']['baseline'] = params
                    st.success("✅ Baseline correction applied!")
                    st.rerun()
            
            else:  # ALS
                col_a, col_b = st.columns(2)
                with col_a:
                    lam = st.number_input(
                        "Lambda (smoothness):",
                        min_value=1e2,
                        max_value=1e9,
                        value=BASELINE_DEFAULTS['als_lambda'],
                        format="%.0e",
                        help="Higher = smoother baseline"
                    )
                with col_b:
                    p = st.number_input(
                        "p (asymmetry):",
                        min_value=0.001,
                        max_value=0.1,
                        value=BASELINE_DEFAULTS['als_p'],
                        format="%.3f",
                        help="Lower = baseline follows valleys"
                    )
                
                if st.button("Apply Baseline Correction", key="baseline_als_btn"):
                    df_corrected, baseline, params = baseline_als_correction(df_raw, lam=lam, p=p)
                    st.session_state.processed_data[st.session_state.current_file]['baseline_corrected'] = df_corrected
                    st.session_state.processed_data[st.session_state.current_file]['baseline'] = baseline
                    st.session_state.processed_data[st.session_state.current_file]['params']['baseline'] = params
                    st.success("✅ Baseline correction applied!")
                    st.rerun()
        
        with col2:
            if st.session_state.processed_data[st.session_state.current_file]['baseline_corrected'] is not None:
                df_corrected = st.session_state.processed_data[st.session_state.current_file]['baseline_corrected']
                baseline = st.session_state.processed_data[st.session_state.current_file]['baseline']
                
                fig = plot_baseline_correction(
                    df_raw, df_corrected, baseline,
                    n_points=n_points if baseline_method == "Linear (Endpoints)" else None,
                    interactive=interactive_plots
                )
                
                if interactive_plots:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.pyplot(fig)
            else:
                st.info("👆 Apply baseline correction to see the results")
        
        st.markdown("---")
        
        # Normalization Section
        st.markdown("### 2️⃣ Normalization")
        
        if st.session_state.processed_data[st.session_state.current_file]['baseline_corrected'] is None:
            st.warning("⚠️ Please apply baseline correction first")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                norm_method = st.selectbox(
                    "Normalization Method:",
                    ["minmax", "max", "area", "vector"],
                    help="Choose normalization algorithm"
                )
                
                with st.expander("ℹ️ Method Info"):
                    if norm_method == "minmax":
                        st.markdown("**Min-Max:** Scales to [0, 1] range")
                    elif norm_method == "max":
                        st.markdown("**Max:** Divides by maximum value")
                    elif norm_method == "area":
                        st.markdown("**Area:** Normalizes by area under curve")
                    else:
                        st.markdown("**Vector:** L2 normalization")
                
                if st.button("Apply Normalization", key="norm_btn"):
                    df_baseline = st.session_state.processed_data[st.session_state.current_file]['baseline_corrected']
                    df_norm, params = normalize_spectrum(df_baseline, method=norm_method)
                    st.session_state.processed_data[st.session_state.current_file]['normalized'] = df_norm
                    st.session_state.processed_data[st.session_state.current_file]['params']['normalization'] = params
                    st.success("✅ Normalization applied!")
                    st.rerun()
            
            with col2:
                if st.session_state.processed_data[st.session_state.current_file]['normalized'] is not None:
                    df_baseline = st.session_state.processed_data[st.session_state.current_file]['baseline_corrected']
                    df_norm = st.session_state.processed_data[st.session_state.current_file]['normalized']
                    
                    fig = plot_normalization_comparison(
                        df_baseline, df_norm, norm_method,
                        interactive=interactive_plots
                    )
                    
                    if interactive_plots:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.pyplot(fig)
                else:
                    st.info("👆 Apply normalization to see the results")
        
        st.markdown("---")
        
        # Optional Smoothing
        st.markdown("### 3️⃣ Smoothing (Optional)")
        
        if st.session_state.processed_data[st.session_state.current_file]['normalized'] is None:
            st.warning("⚠️ Please complete baseline correction and normalization first")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                apply_smoothing = st.checkbox("Apply Savitzky-Golay smoothing")
                
                if apply_smoothing:
                    window_length = st.slider(
                        "Window length:",
                        min_value=5,
                        max_value=51,
                        value=11,
                        step=2,
                        help="Must be odd number"
                    )
                    
                    polyorder = st.slider(
                        "Polynomial order:",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Degree of polynomial"
                    )
                    
                    if st.button("Apply Smoothing", key="smooth_btn"):
                        df_norm = st.session_state.processed_data[st.session_state.current_file]['normalized']
                        df_smooth, params = smooth_spectrum(df_norm, window_length=window_length, 
                                                           polyorder=polyorder)
                        st.session_state.processed_data[st.session_state.current_file]['smoothed'] = df_smooth
                        st.session_state.processed_data[st.session_state.current_file]['params']['smoothing'] = params
                        st.success("✅ Smoothing applied!")
                        st.rerun()
            
            with col2:
                if apply_smoothing and st.session_state.processed_data[st.session_state.current_file]['smoothed'] is not None:
                    df_norm = st.session_state.processed_data[st.session_state.current_file]['normalized']
                    df_smooth = st.session_state.processed_data[st.session_state.current_file]['smoothed']
                    
                    fig = plot_normalization_comparison(
                        df_norm, df_smooth, "Savitzky-Golay",
                        interactive=interactive_plots
                    )
                    
                    if interactive_plots:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.pyplot(fig)
    
    # TAB 3: Peak Detection
    with tab3:
        st.markdown('<h2 class="subheader">🔍 Peak Detection</h2>', unsafe_allow_html=True)
        
        # Get processed data
        processed = st.session_state.processed_data[st.session_state.current_file]
        
        if processed['normalized'] is None:
            st.warning("⚠️ Please complete preprocessing (baseline correction and normalization) first")
        else:
            # Use smoothed if available, otherwise normalized
            df_for_peaks = processed['smoothed'] if processed['smoothed'] is not None else processed['normalized']
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("#### Detection Parameters")
                
                height = st.slider(
                    "Minimum height:",
                    min_value=0.0,
                    max_value=1.0,
                    value=PEAK_DETECTION_DEFAULTS['height'],
                    step=0.01,
                    help="Minimum peak height threshold"
                )
                
                prominence = st.slider(
                    "Minimum prominence:",
                    min_value=0.0,
                    max_value=0.5,
                    value=PEAK_DETECTION_DEFAULTS['prominence'],
                    step=0.01,
                    help="How much peak stands out"
                )
                
                distance = st.slider(
                    "Minimum distance:",
                    min_value=10,
                    max_value=100,
                    value=PEAK_DETECTION_DEFAULTS['distance'],
                    step=5,
                    help="Minimum spacing between peaks (data points)"
                )
                
                n_peaks = st.number_input(
                    "Number of peaks:",
                    min_value=1,
                    max_value=50,
                    value=PEAK_DETECTION_DEFAULTS['n_peaks'],
                    help="Number of top peaks to detect"
                )
                
                if st.button("🔍 Detect Peaks", key="detect_peaks_btn", type="primary"):
                    peaks, peak_df, params = detect_peaks(
                        df_for_peaks,
                        height=height,
                        prominence=prominence,
                        distance=distance,
                        n_peaks=n_peaks
                    )
                    
                    st.session_state.processed_data[st.session_state.current_file]['peaks'] = peaks
                    st.session_state.processed_data[st.session_state.current_file]['peak_df'] = peak_df
                    st.session_state.processed_data[st.session_state.current_file]['params']['peak_detection'] = params
                    
                    st.success(f"✅ Detected {len(peaks)} peaks!")
                    st.rerun()
                
                # Show detection info
                if 'peaks' in processed and processed['peaks'] is not None:
                    st.markdown("---")
                    st.markdown("#### Detection Results")
                    params = processed['params']['peak_detection']
                    st.metric("Peaks Found", params['n_peaks_found'])
                    st.metric("Total Peaks", params['n_peaks_total'])
            
            with col2:
                if 'peaks' in processed and processed['peaks'] is not None:
                    peaks = processed['peaks']
                    peak_df = processed['peak_df']
                    
                    # Plot
                    fig = plot_peak_detection(
                        df_for_peaks, peaks, peak_df,
                        interactive=interactive_plots
                    )
                    
                    if interactive_plots:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.pyplot(fig)
                    
                    # Peak table
                    st.markdown("#### Detected Peaks")
                    st.dataframe(peak_df, use_container_width=True, hide_index=True)
                    
                    # Categorization
                    st.markdown("#### Peak Categorization")
                    categorized = categorize_peaks(peak_df, CNT_BANDS)
                    
                    for band_name, band_data in categorized.items():
                        with st.expander(f"{band_name} ({band_data['count']} peaks)"):
                            st.markdown(f"**Description:** {band_data['description']}")
                            st.dataframe(band_data['peaks'][['Peak_Number', 'Wavenumber', 'Intensity']], 
                                       use_container_width=True, hide_index=True)
                    
                    # Statistics
                    st.markdown("#### Peak Statistics")
                    stats = get_peak_statistics(peak_df)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Mean Intensity", f"{stats['mean_intensity']:.3f}")
                    with col_b:
                        st.metric("Std Intensity", f"{stats['std_intensity']:.3f}")
                    with col_c:
                        st.metric("Strongest Peak", f"{stats['strongest_peak_wavenumber']:.0f} cm⁻¹")
                
                else:
                    st.info("👈 Adjust parameters and click 'Detect Peaks'")
    
    # TAB 4: Peak Fitting
    with tab4:
        st.markdown('<h2 class="subheader">📈 Peak Fitting & Deconvolution</h2>', unsafe_allow_html=True)
        
        processed = st.session_state.processed_data[st.session_state.current_file]
        
        if processed['normalized'] is None:
            st.warning("⚠️ Please complete preprocessing first")
        else:
            df_for_fitting = processed['smoothed'] if processed['smoothed'] is not None else processed['normalized']
            
            # Initialize fitting results in session state
            if 'fitting_results' not in processed:
                processed['fitting_results'] = {}
            
            st.markdown("### Select Region to Fit")
            
            # Region selector
            col1, col2 = st.columns([1, 3])
            
            with col1:
                region_preset = st.selectbox(
                    "Preset Region:",
                    ["Custom", "D-band (1200-1450)", "G-band (1450-1800)", 
                     "D+G bands (1200-1800)", "2D-band (2500-3000)"],
                    help="Choose a preset or define custom range"
                )
                
                if region_preset == "Custom":
                    min_wn = st.number_input(
                        "Min Wavenumber:",
                        min_value=float(df_for_fitting['Wavenumber'].min()),
                        max_value=float(df_for_fitting['Wavenumber'].max()),
                        value=1200.0
                    )
                    max_wn = st.number_input(
                        "Max Wavenumber:",
                        min_value=float(df_for_fitting['Wavenumber'].min()),
                        max_value=float(df_for_fitting['Wavenumber'].max()),
                        value=1800.0
                    )
                    region_range = (min_wn, max_wn)
                    region_name = f"Custom_{int(min_wn)}-{int(max_wn)}"
                
                elif region_preset == "D-band (1200-1450)":
                    region_range = PEAK_FITTING_DEFAULTS['d_band_range']
                    region_name = "D_band"
                elif region_preset == "G-band (1450-1800)":
                    region_range = PEAK_FITTING_DEFAULTS['g_band_range']
                    region_name = "G_band"
                elif region_preset == "D+G bands (1200-1800)":
                    region_range = (1200, 1800)
                    region_name = "DG_bands"
                else:  # 2D-band
                    region_range = PEAK_FITTING_DEFAULTS['2d_band_range']
                    region_name = "2D_band"
                
                st.markdown("---")
                
                st.markdown("#### Fitting Parameters")
                
                n_peaks_fit = st.number_input(
                    "Number of peaks:",
                    min_value=1,
                    max_value=10,
                    value=2 if "DG" in region_name or region_preset == "D+G bands (1200-1800)" else 1,
                    help="Number of peaks to fit in this region"
                )
                
                peak_type = st.selectbox(
                    "Peak Model:",
                    ["lorentzian", "gaussian"],
                    help="Lorentzian is typical for Raman"
                )
                
                # Advanced options
                with st.expander("🔧 Advanced Options"):
                    max_iterations = st.number_input(
                        "Max iterations:",
                        min_value=1000,
                        max_value=50000,
                        value=PEAK_FITTING_DEFAULTS['max_iterations']
                    )
                    
                    use_auto_guess = st.checkbox(
                        "Auto-detect initial centers",
                        value=True,
                        help="Automatically find peak positions for initial guess"
                    )
                    
                    if not use_auto_guess:
                        initial_centers_str = st.text_input(
                            "Initial centers (comma-separated):",
                            value="1350, 1580" if n_peaks_fit == 2 else "1350",
                            help="Example: 1350, 1580"
                        )
                        try:
                            initial_centers = [float(x.strip()) for x in initial_centers_str.split(',')]
                        except:
                            initial_centers = None
                            st.error("Invalid format for initial centers")
                    else:
                        initial_centers = None
                
                if st.button("📊 Fit Peaks", key=f"fit_{region_name}_btn", type="primary"):
                    with st.spinner("Fitting peaks..."):
                        fit_result = fit_peaks_region(
                            df_for_fitting,
                            region_range=region_range,
                            n_peaks=n_peaks_fit,
                            peak_type=peak_type,
                            initial_centers=initial_centers,
                            max_iterations=max_iterations
                        )
                        
                        processed['fitting_results'][region_name] = fit_result
                        
                        if fit_result['success']:
                            st.success(f"✅ Fitting successful! R² = {fit_result['r_squared']:.4f}")
                            st.rerun()
                        else:
                            st.error(f"❌ Fitting failed: {fit_result['error']}")
            
            with col2:
                if region_name in processed['fitting_results']:
                    fit_result = processed['fitting_results'][region_name]
                    
                    if fit_result['success']:
                        # Plot fitting
                        fig = plot_peak_fitting(fit_result, interactive=interactive_plots)
                        
                        if fig is not None:
                            if interactive_plots:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.pyplot(fig)
                        
                        # Results table
                        st.markdown("#### Fitted Parameters")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("R² Score", f"{fit_result['r_squared']:.4f}")
                        with col_b:
                            st.metric("RMSE", f"{fit_result['rmse']:.4f}")
                        with col_c:
                            st.metric("Peak Model", fit_result['peak_type'].capitalize())
                        
                        st.dataframe(fit_result['peak_info'], use_container_width=True, hide_index=True)
                        
                        # Uncertainties if available
                        if fit_result['uncertainties'] is not None:
                            with st.expander("📊 Parameter Uncertainties"):
                                st.dataframe(fit_result['uncertainties'], use_container_width=True, hide_index=True)
                    
                else:
                    st.info("👈 Select region and click 'Fit Peaks'")
            
            # Material Analysis Section
            st.markdown("---")
            st.markdown("### 🔬 Material Characterization")
            
            # Check if D and G bands are fitted
            has_d_band = 'D_band' in processed['fitting_results'] and processed['fitting_results']['D_band']['success']
            has_g_band = 'G_band' in processed['fitting_results'] and processed['fitting_results']['G_band']['success']
            has_dg_bands = 'DG_bands' in processed['fitting_results'] and processed['fitting_results']['DG_bands']['success']
            has_2d_band = '2D_band' in processed['fitting_results'] and processed['fitting_results']['2D_band']['success']
            
            if has_dg_bands or (has_d_band and has_g_band):
                st.markdown("#### ID/IG Ratio Analysis")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    ratio_method = st.radio(
                        "Calculation method:",
                        ["area", "height"],
                        help="Use peak area (more accurate) or peak height"
                    )
                    
                    if st.button("Calculate ID/IG Ratio", key="calc_id_ig_btn"):
                        if has_dg_bands:
                            # Extract D and G from combined fit
                            dg_result = processed['fitting_results']['DG_bands']
                            peak_info = dg_result['peak_info']
                            
                            # Assume first peak is D, second is G
                            d_peak = peak_info.iloc[0] if peak_info.iloc[0]['Center'] < 1500 else peak_info.iloc[1]
                            g_peak = peak_info.iloc[1] if peak_info.iloc[0]['Center'] < 1500 else peak_info.iloc[0]
                            
                            if ratio_method == 'area':
                                ratio = d_peak['Area'] / g_peak['Area']
                            else:
                                ratio = d_peak['Amplitude'] / g_peak['Amplitude']
                            
                            # Create pseudo fit results
                            id_ig_result = {
                                'success': True,
                                'ratio': ratio,
                                'method': ratio_method,
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
                                'crystallite_size_nm': 4.4 / ratio if ratio > 0 else None,
                                'quality_assessment': (
                                    "High quality (low defects)" if ratio < 0.8 else
                                    "Good quality (moderate defects)" if ratio < 1.2 else
                                    "Lower quality or highly functionalized (high defects)"
                                )
                            }
                        else:
                            d_fit = processed['fitting_results']['D_band']
                            g_fit = processed['fitting_results']['G_band']
                            id_ig_result = calculate_id_ig_ratio(d_fit, g_fit, method=ratio_method)
                        
                        processed['id_ig_analysis'] = id_ig_result
                        st.success("✅ ID/IG ratio calculated!")
                        st.rerun()
                
                with col2:
                    if 'id_ig_analysis' in processed and processed['id_ig_analysis']['success']:
                        result = processed['id_ig_analysis']
                        
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>Analysis Results</h4>
                        <p><strong>ID/IG Ratio:</strong> {result['ratio']:.3f} ({result['method']})</p>
                        <p><strong>D-band position:</strong> {result['d_band']['center']:.1f} cm⁻¹ 
                           (FWHM: {result['d_band']['fwhm']:.1f} cm⁻¹)</p>
                        <p><strong>G-band position:</strong> {result['g_band']['center']:.1f} cm⁻¹ 
                           (FWHM: {result['g_band']['fwhm']:.1f} cm⁻¹)</p>
                        <p><strong>Estimated crystallite size (La):</strong> {result['crystallite_size_nm']:.1f} nm</p>
                        <p><strong>Quality Assessment:</strong> {result['quality_assessment']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("ID/IG Ratio", f"{result['ratio']:.3f}")
                        with col_b:
                            st.metric("La (nm)", f"{result['crystallite_size_nm']:.1f}")
                        with col_c:
                            st.metric("D-band FWHM", f"{result['d_band']['fwhm']:.1f}")
            
            else:
                st.info("💡 Fit D and G band regions to calculate ID/IG ratio")
            
            # 2D/G ratio analysis
            if has_2d_band and (has_g_band or has_dg_bands):
                st.markdown("---")
                st.markdown("#### I2D/IG Ratio Analysis")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if st.button("Calculate I2D/IG Ratio", key="calc_i2d_ig_btn"):
                        fit_2d = processed['fitting_results']['2D_band']
                        
                        if has_dg_bands:
                            dg_result = processed['fitting_results']['DG_bands']
                            peak_info = dg_result['peak_info']
                            g_peak = peak_info.iloc[1] if peak_info.iloc[0]['Center'] < 1500 else peak_info.iloc[0]
                            
                            # Create pseudo G-band fit result
                            fit_g = {
                                'success': True,
                                'peak_info': pd.DataFrame([{
                                    'Peak': 1,
                                    'Center': g_peak['Center'],
                                    'Amplitude': g_peak['Amplitude'],
                                    'Width': g_peak['Width'],
                                    'FWHM': g_peak['FWHM'],
                                    'Area': g_peak['Area']
                                }])
                            }
                        else:
                            fit_g = processed['fitting_results']['G_band']
                        
                        i2d_ig_result = calculate_i2d_ig_ratio(fit_2d, fit_g)
                        processed['i2d_ig_analysis'] = i2d_ig_result
                        st.success("✅ I2D/IG ratio calculated!")
                        st.rerun()
                
                with col2:
                    if 'i2d_ig_analysis' in processed and processed['i2d_ig_analysis']['success']:
                        result = processed['i2d_ig_analysis']
                        
                        st.markdown(f"""
                        <div class="info-box">
                        <h4>I2D/IG Analysis</h4>
                        <p><strong>I2D/IG Ratio (height):</strong> {result['ratio_height']:.3f}</p>
                        <p><strong>I2D/IG Ratio (area):</strong> {result['ratio_area']:.3f}</p>
                        <p><strong>2D-band position:</strong> {result['2d_band']['center']:.1f} cm⁻¹ 
                           (FWHM: {result['2d_band']['fwhm']:.1f} cm⁻¹)</p>
                        <p><strong>Layer Estimate:</strong> {result['layer_estimate']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            else:
                if has_2d_band and not (has_g_band or has_dg_bands):
                    st.info("💡 Fit G-band region to calculate I2D/IG ratio")
                elif not has_2d_band and (has_g_band or has_dg_bands):
                    st.info("💡 Fit 2D-band region to calculate I2D/IG ratio")
    
    # TAB 5: Analysis & Results
    with tab5:
        st.markdown('<h2 class="subheader">📋 Analysis Summary & Results</h2>', unsafe_allow_html=True)
        
        processed = st.session_state.processed_data[st.session_state.current_file]
        
        # Processing Summary
        st.markdown("### 📊 Processing Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Preprocessing")
            if processed['baseline_corrected'] is not None:
                st.success("✅ Baseline corrected")
                params = processed['params'].get('baseline', {})
                st.caption(f"Method: {params.get('method', 'N/A')}")
            else:
                st.warning("⚠️ Not applied")
            
            if processed['normalized'] is not None:
                st.success("✅ Normalized")
                params = processed['params'].get('normalization', {})
                st.caption(f"Method: {params.get('method', 'N/A')}")
            else:
                st.warning("⚠️ Not applied")
            
            if processed['smoothed'] is not None:
                st.success("✅ Smoothed")
                params = processed['params'].get('smoothing', {})
                st.caption(f"Window: {params.get('window_length', 'N/A')}")
            else:
                st.info("ℹ️ Optional - not applied")
        
        with col2:
            st.markdown("#### Peak Detection")
            if 'peaks' in processed and processed['peaks'] is not None:
                st.success(f"✅ {len(processed['peaks'])} peaks detected")
                params = processed['params'].get('peak_detection', {})
                st.caption(f"Height: {params.get('height', 'N/A')}")
                st.caption(f"Prominence: {params.get('prominence', 'N/A')}")
            else:
                st.warning("⚠️ Not performed")
        
        with col3:
            st.markdown("#### Peak Fitting")
            if 'fitting_results' in processed:
                n_fitted = sum(1 for r in processed['fitting_results'].values() if r['success'])
                if n_fitted > 0:
                    st.success(f"✅ {n_fitted} region(s) fitted")
                    for name, result in processed['fitting_results'].items():
                        if result['success']:
                            st.caption(f"{name}: R² = {result['r_squared']:.3f}")
                else:
                    st.warning("⚠️ No successful fits")
            else:
                st.warning("⚠️ Not performed")
        
        st.markdown("---")
        
        # Material Characterization Results
        st.markdown("### 🔬 Material Characterization")
        
        if 'id_ig_analysis' in processed and processed['id_ig_analysis']['success']:
            result = processed['id_ig_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### CNT Quality Metrics")
                
                metrics_df = pd.DataFrame({
                    'Parameter': [
                        'ID/IG Ratio',
                        'D-band Position',
                        'G-band Position',
                        'D-band FWHM',
                        'G-band FWHM',
                        'Crystallite Size (La)'
                    ],
                    'Value': [
                        f"{result['ratio']:.3f}",
                        f"{result['d_band']['center']:.1f} cm⁻¹",
                        f"{result['g_band']['center']:.1f} cm⁻¹",
                        f"{result['d_band']['fwhm']:.1f} cm⁻¹",
                        f"{result['g_band']['fwhm']:.1f} cm⁻¹",
                        f"{result['crystallite_size_nm']:.1f} nm"
                    ]
                })
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Quality Assessment")
                st.markdown(f"""
                <div class="{'success-box' if result['ratio'] < 0.8 else 'warning-box' if result['ratio'] < 1.2 else 'info-box'}">
                <h4>{result['quality_assessment']}</h4>
                <p><strong>ID/IG = {result['ratio']:.3f}</strong></p>
                <ul>
                    <li>Lower ID/IG → fewer defects → better quality</li>
                    <li>Higher ID/IG → more defects → lower quality or functionalized</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation guide
                with st.expander("📖 Interpretation Guide"):
                    st.markdown("""
                    **ID/IG Ratio Interpretation:**
                    - **< 0.8**: High quality CNT with minimal defects
                    - **0.8 - 1.2**: Good quality CNT with moderate defects
                    - **> 1.2**: Lower quality or highly functionalized CNT
                    
                    **D-band (~1350 cm⁻¹)**: Defects and disorder in graphitic structure
                    
                    **G-band (~1580 cm⁻¹)**: In-plane vibrations of sp² carbon
                    
                    **Crystallite Size (La)**: Average size of ordered crystalline domains
                    """)
        
        else:
            st.info("💡 Complete peak fitting for D and G bands to see material characterization")
        
        # 2D/G analysis if available
        if 'i2d_ig_analysis' in processed and processed['i2d_ig_analysis']['success']:
            st.markdown("---")
            result = processed['i2d_ig_analysis']
            
            st.markdown("#### Layer Structure Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_df = pd.DataFrame({
                    'Parameter': [
                        'I2D/IG Ratio (height)',
                        'I2D/IG Ratio (area)',
                        '2D-band Position',
                        '2D-band FWHM'
                    ],
                    'Value': [
                        f"{result['ratio_height']:.3f}",
                        f"{result['ratio_area']:.3f}",
                        f"{result['2d_band']['center']:.1f} cm⁻¹",
                        f"{result['2d_band']['fwhm']:.1f} cm⁻¹"
                    ]
                })
                
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-box">
                <h4>Layer Estimation</h4>
                <p><strong>{result['layer_estimate']}</strong></p>
                <ul>
                    <li>I2D/IG > 2: Monolayer</li>
                    <li>I2D/IG 1-2: Bilayer</li>
                    <li>I2D/IG 0.5-1: Few-layer (3-5)</li>
                    <li>I2D/IG < 0.5: Multilayer/CNT</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Complete Results Table
        st.markdown("### 📄 Complete Results Table")
        
        if 'peak_df' in processed and processed['peak_df'] is not None:
            st.markdown("#### All Detected Peaks")
            st.dataframe(processed['peak_df'], use_container_width=True, hide_index=True)
        
        if 'fitting_results' in processed and len(processed['fitting_results']) > 0:
            st.markdown("#### Fitted Peak Parameters")
            
            for region_name, fit_result in processed['fitting_results'].items():
                if fit_result['success']:
                    with st.expander(f"📊 {region_name} - R² = {fit_result['r_squared']:.4f}"):
                        st.dataframe(fit_result['peak_info'], use_container_width=True, hide_index=True)
    
    # TAB 6: Export
    with tab6:
        st.markdown('<h2 class="subheader">💾 Export Results</h2>', unsafe_allow_html=True)
        
        processed = st.session_state.processed_data[st.session_state.current_file]
        
        st.markdown("### 📊 Export Processed Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Select Data to Export")
            
            export_options = []
            if processed['baseline_corrected'] is not None:
                export_options.append("Baseline Corrected")
            if processed['normalized'] is not None:
                export_options.append("Normalized")
            if processed['smoothed'] is not None:
                export_options.append("Smoothed")
            
            if len(export_options) > 0:
                data_to_export = st.selectbox("Choose dataset:", export_options)
                
                export_format = st.selectbox("Export format:", ["CSV", "Excel", "JSON"])
                
                # Get the selected data
                if data_to_export == "Baseline Corrected":
                    df_export = processed['baseline_corrected']
                elif data_to_export == "Normalized":
                    df_export = processed['normalized']
                else:
                    df_export = processed['smoothed']
                
                filename = f"{st.session_state.current_file.replace('.txt', '')}_{data_to_export.replace(' ', '_')}"
                
                if st.button("📥 Download Processed Spectrum", key="export_spectrum_btn"):
                    file_data = export_spectrum(df_export, filename, format=export_format.lower())
                    
                    st.download_button(
                        label=f"Download {export_format}",
                        data=file_data,
                        file_name=f"{filename}.{export_format.lower()}",
                        mime=f"application/{export_format.lower()}"
                    )
            else:
                st.info("No processed data available for export")
        
        with col2:
            st.markdown("#### Export Peak Information")
            
            if 'peak_df' in processed and processed['peak_df'] is not None:
                peak_format = st.selectbox("Peak data format:", ["CSV", "Excel"], key="peak_format")
                
                if st.button("📥 Download Peak List", key="export_peaks_btn"):
                    peak_data = export_spectrum(processed['peak_df'], "peaks", format=peak_format.lower())
                    
                    st.download_button(
                        label=f"Download Peaks ({peak_format})",
                        data=peak_data,
                        file_name=f"{st.session_state.current_file.replace('.txt', '')}_peaks.{peak_format.lower()}",
                        mime=f"application/{peak_format.lower()}"
                    )
            else:
                st.info("No peak data available")
        
        st.markdown("---")
        
        # Export fitting results
        if 'fitting_results' in processed and len(processed['fitting_results']) > 0:
            st.markdown("### 📈 Export Fitting Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Combine all fitting results
                all_fitting_data = []
                for region_name, fit_result in processed['fitting_results'].items():
                    if fit_result['success']:
                        peak_info = fit_result['peak_info'].copy()
                        peak_info['Region'] = region_name
                        peak_info['R_squared'] = fit_result['r_squared']
                        peak_info['Peak_Type'] = fit_result['peak_type']
                        all_fitting_data.append(peak_info)
                
                if len(all_fitting_data) > 0:
                    combined_fitting = pd.concat(all_fitting_data, ignore_index=True)
                    
                    fitting_format = st.selectbox("Fitting results format:", ["CSV", "Excel"], key="fitting_format")
                    
                    if st.button("📥 Download Fitting Results", key="export_fitting_btn"):
                        fitting_data = export_spectrum(combined_fitting, "fitting_results", 
                                                      format=fitting_format.lower())
                        
                        st.download_button(
                            label=f"Download Fitting Results ({fitting_format})",
                            data=fitting_data,
                            file_name=f"{st.session_state.current_file.replace('.txt', '')}_fitting.{fitting_format.lower()}",
                            mime=f"application/{fitting_format.lower()}"
                        )
            
            with col2:
                # Material characterization summary
                if 'id_ig_analysis' in processed and processed['id_ig_analysis']['success']:
                    st.markdown("#### Material Characterization Report")
                    
                    result = processed['id_ig_analysis']
                    
                    report_data = pd.DataFrame({
                        'Parameter': [
                            'Sample Name',
                            'ID/IG Ratio',
                            'D-band Position (cm⁻¹)',
                            'G-band Position (cm⁻¹)',
                            'D-band FWHM (cm⁻¹)',
                            'G-band FWHM (cm⁻¹)',
                            'Crystallite Size La (nm)',
                            'Quality Assessment'
                        ],
                        'Value': [
                            st.session_state.current_file,
                            f"{result['ratio']:.3f}",
                            f"{result['d_band']['center']:.1f}",
                            f"{result['g_band']['center']:.1f}",
                            f"{result['d_band']['fwhm']:.1f}",
                            f"{result['g_band']['fwhm']:.1f}",
                            f"{result['crystallite_size_nm']:.1f}",
                            result['quality_assessment']
                        ]
                    })
                    
                    report_format = st.selectbox("Report format:", ["CSV", "Excel"], key="report_format")
                    
                    if st.button("📥 Download Characterization Report", key="export_report_btn"):
                        report_file = export_spectrum(report_data, "characterization", 
                                                     format=report_format.lower())
                        
                        st.download_button(
                            label=f"Download Report ({report_format})",
                            data=report_file,
                            file_name=f"{st.session_state.current_file.replace('.txt', '')}_report.{report_format.lower()}",
                            mime=f"application/{report_format.lower()}"
                        )
        
        st.markdown("---")
        
        # Batch export for multiple files
        if len(st.session_state.uploaded_files) > 1:
            st.markdown("### 📦 Batch Export (All Files)")
            
            st.info("Export summary data for all loaded spectra")
            
            if st.button("📥 Generate Batch Summary", key="batch_export_btn"):
                # Collect data from all files
                batch_summary = []
                
                for filename, file_data in st.session_state.uploaded_files.items():
                    if filename in st.session_state.processed_data:
                        proc = st.session_state.processed_data[filename]
                        
                        summary_row = {
                            'Filename': filename,
                            'N_Points': file_data['metadata']['n_points'],
                            'Wavenumber_Range': f"{file_data['metadata']['wavenumber_range'][0]:.1f}-{file_data['metadata']['wavenumber_range'][1]:.1f}"
                        }
                        
                        if 'id_ig_analysis' in proc and proc['id_ig_analysis']['success']:
                            result = proc['id_ig_analysis']
                            summary_row.update({
                                'ID_IG_Ratio': result['ratio'],
                                'D_band_Position': result['d_band']['center'],
                                'G_band_Position': result['g_band']['center'],
                                'Crystallite_Size_nm': result['crystallite_size_nm'],
                                'Quality': result['quality_assessment']
                            })
                        
                        if 'peaks' in proc:
                            summary_row['N_Peaks_Detected'] = len(proc['peaks'])
                        
                        batch_summary.append(summary_row)
                
                if len(batch_summary) > 0:
                    batch_df = pd.DataFrame(batch_summary)
                    
                    batch_format = st.selectbox("Batch summary format:", ["CSV", "Excel"], key="batch_format")
                    
                    batch_data = export_spectrum(batch_df, "batch_summary", format=batch_format.lower())
                    
                    st.download_button(
                        label=f"Download Batch Summary ({batch_format})",
                        data=batch_data,
                        file_name=f"batch_summary.{batch_format.lower()}",
                        mime=f"application/{batch_format.lower()}"
                    )
                    
                    st.dataframe(batch_df, use_container_width=True)
                else:
                    st.warning("No processed data available for batch export")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 2rem;">
    <p>Raman Spectroscopy Analyzer v1.0 | Built with Streamlit</p>
    <p>For CNT, graphene, and carbon nanomaterial characterization</p>
</div>
""", unsafe_allow_html=True)
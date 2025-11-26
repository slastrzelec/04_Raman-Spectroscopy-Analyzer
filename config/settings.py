"""
Configuration settings for Raman Spectroscopy Analysis
Default parameters and constants
"""

# =============================================================================
# BASELINE CORRECTION DEFAULTS
# =============================================================================

BASELINE_DEFAULTS = {
    # Linear endpoint method
    'linear_n_points': 5,
    
    # Asymmetric Least Squares (ALS) method
    'als_lambda': 1e6,      # Smoothness parameter (1e2 to 1e9)
    'als_p': 0.01,          # Asymmetry parameter (0.001 to 0.1)
    'als_iterations': 10,   # Number of iterations
    
    # Polynomial baseline
    'polynomial_degree': 3,
    
    # Rolling ball method
    'rolling_ball_window': 100
}

# =============================================================================
# NORMALIZATION DEFAULTS
# =============================================================================

NORMALIZATION_DEFAULT = 'minmax'

NORMALIZATION_METHODS = {
    'minmax': {
        'name': 'Min-Max',
        'description': 'Scales intensity to [0, 1] range',
        'formula': '(I - I_min) / (I_max - I_min)',
        'use_case': 'General comparison and visualization'
    },
    'max': {
        'name': 'Maximum',
        'description': 'Divides by maximum intensity',
        'formula': 'I / I_max',
        'use_case': 'Peak ratio analysis (ID/IG)'
    },
    'area': {
        'name': 'Area',
        'description': 'Normalizes by area under curve',
        'formula': 'I / ∫I dν',
        'use_case': 'Quantitative analysis'
    },
    'vector': {
        'name': 'Vector (L2)',
        'description': 'L2 normalization to unit length',
        'formula': 'I / ||I||₂',
        'use_case': 'Machine learning applications'
    }
}

# =============================================================================
# SMOOTHING DEFAULTS
# =============================================================================

SMOOTHING_DEFAULTS = {
    'savgol_window_length': 11,  # Must be odd
    'savgol_polyorder': 3,
    'gaussian_sigma': 2.0
}

# =============================================================================
# PEAK DETECTION DEFAULTS
# =============================================================================

PEAK_DETECTION_DEFAULTS = {
    'height': 0.15,           # Minimum peak height (0.0 to 1.0 for normalized)
    'prominence': 0.08,       # Minimum peak prominence
    'distance': 50,           # Minimum distance between peaks (data points)
    'n_peaks': 11,            # Number of top peaks to detect
    'width': None,            # Minimum peak width (None = no constraint)
    'rel_height': 0.5         # Relative height for width calculation
}

# =============================================================================
# PEAK FITTING DEFAULTS
# =============================================================================

PEAK_FITTING_DEFAULTS = {
    # CNT band regions (cm⁻¹)
    'd_band_range': (1200, 1450),
    'g_band_range': (1450, 1800),
    '2d_band_range': (2500, 3000),
    'd_plus_g_range': (1200, 1800),
    
    # Fitting parameters
    'peak_type': 'lorentzian',  # 'lorentzian', 'gaussian', or 'voigt'
    'max_iterations': 10000,
    
    # Initial parameter bounds
    'amplitude_bounds': (0.0, 2.0),
    'center_bounds': 'auto',  # or tuple (min, max)
    'width_bounds': (1.0, 100.0),
    
    # Number of peaks per region
    'n_peaks_d_band': 1,
    'n_peaks_g_band': 1,
    'n_peaks_2d_band': 1,
    'n_peaks_d_plus_g': 2
}

# =============================================================================
# CNT/GRAPHENE CHARACTERISTIC BANDS
# =============================================================================

CNT_BANDS = {
    'D-band': {
        'range': (1200, 1450),
        'typical': 1350,
        'description': 'Defects/disorder in graphitic structure',
        'full_name': 'D (Disorder) band',
        'origin': 'Breathing mode of sp² carbon rings',
        'symmetry': 'A₁g'
    },
    'G-band': {
        'range': (1450, 1800),
        'typical': 1580,
        'description': 'In-plane vibrations of sp² carbon',
        'full_name': 'G (Graphitic) band',
        'origin': 'In-plane stretching of C-C bonds',
        'symmetry': 'E₂g'
    },
    'D\'-band': {
        'range': (1580, 1620),
        'typical': 1620,
        'description': 'Defect-induced disorder',
        'full_name': 'D\' (D-prime) band',
        'origin': 'Intervalley double resonance process',
        'symmetry': 'A₁g'
    },
    '2D-band': {
        'range': (2500, 3000),
        'typical': 2700,
        'description': 'Second-order two-phonon process',
        'full_name': '2D (G\') band',
        'origin': 'Two-phonon double resonance',
        'symmetry': 'A₁g'
    },
    'D+G-band': {
        'range': (2900, 3100),
        'typical': 2950,
        'description': 'Combination mode',
        'full_name': 'D+G combination band',
        'origin': 'Combination of D and G phonons',
        'symmetry': None
    },
    '2D\'-band': {
        'range': (3150, 3250),
        'typical': 3200,
        'description': 'Second-order D\' mode',
        'full_name': '2D\' band',
        'origin': 'Overtone of D\' band',
        'symmetry': None
    }
}

# =============================================================================
# MATERIAL QUALITY THRESHOLDS
# =============================================================================

QUALITY_THRESHOLDS = {
    'id_ig_ratio': {
        'high_quality': 0.8,      # ID/IG < 0.8 = high quality
        'good_quality': 1.2,      # ID/IG < 1.2 = good quality
        'low_quality': 1.2        # ID/IG >= 1.2 = lower quality/functionalized
    },
    'i2d_ig_ratio': {
        'monolayer': 2.0,         # I2D/IG > 2 = monolayer graphene
        'bilayer': 1.0,           # I2D/IG 1-2 = bilayer
        'few_layer': 0.5,         # I2D/IG 0.5-1 = few-layer (3-5)
        'multilayer': 0.5         # I2D/IG < 0.5 = multilayer/CNT
    },
    'fwhm_d_band': {
        'sharp': 40,              # FWHM < 40 cm⁻¹ = well-defined
        'broad': 80               # FWHM > 80 cm⁻¹ = broad/disordered
    },
    'fwhm_g_band': {
        'sharp': 15,              # FWHM < 15 cm⁻¹ = crystalline
        'broad': 30               # FWHM > 30 cm⁻¹ = amorphous
    }
}

# =============================================================================
# TUINSTRA-KOENIG RELATION CONSTANTS
# =============================================================================
# La (nm) = C_λ / (ID/IG)
# Crystallite size calculation constants for different laser wavelengths

TUINSTRA_KOENIG_CONSTANTS = {
    514: 4.4,   # 514 nm (Ar+ laser)
    532: 4.4,   # 532 nm (Nd:YAG laser)
    633: 2.4,   # 633 nm (He-Ne laser)
    785: 1.8    # 785 nm (NIR laser)
}

DEFAULT_LASER_WAVELENGTH = 532  # nm

# =============================================================================
# PLOT STYLING
# =============================================================================

PLOT_CONFIG = {
    'figure_width': 14,
    'figure_height': 6,
    'line_width': 1.5,
    'marker_size': 10,
    'font_size': 12,
    'title_size': 14,
    'label_size': 12,
    'legend_size': 10,
    'dpi': 100,
    'transparent_background': False
}

PLOT_COLORS = {
    'spectrum': '#1f77b4',        # Blue
    'baseline': '#d62728',        # Red
    'peaks': '#ff7f0e',           # Orange
    'fitted': '#2ca02c',          # Green
    'residuals': '#9467bd',       # Purple
    'd_band': '#e377c2',          # Pink
    'g_band': '#7f7f7f',          # Gray
    '2d_band': '#bcbd22',         # Yellow-green
    'grid': 'lightgray'
}

PLOTLY_TEMPLATE = 'plotly_white'  # 'plotly', 'plotly_white', 'plotly_dark', 'seaborn'

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

EXPORT_FORMATS = {
    'data': ['CSV', 'Excel', 'JSON'],
    'images': ['PNG', 'PDF', 'SVG', 'JPG'],
    'reports': ['PDF', 'HTML', 'Markdown']
}

EXPORT_DEFAULTS = {
    'data_format': 'CSV',
    'image_format': 'PNG',
    'image_dpi': 300,
    'decimal_places': 4,
    'include_metadata': True,
    'include_processing_params': True
}

# =============================================================================
# FILE VALIDATION
# =============================================================================

VALIDATION_RULES = {
    'min_data_points': 100,
    'max_data_points': 100000,
    'min_wavenumber_range': 100,  # cm⁻¹
    'max_intensity_ratio': 1e6,   # max/min intensity
    'required_columns': ['Wavenumber', 'Intensity'],
    'allow_negative_intensity': False,
    'check_monotonic': True
}

# =============================================================================
# PROCESSING PIPELINE PRESETS
# =============================================================================

PROCESSING_PRESETS = {
    'quick': {
        'name': 'Quick Analysis',
        'description': 'Fast processing with default parameters',
        'baseline': {'method': 'linear', 'n_points': 5},
        'normalization': 'minmax',
        'smoothing': None,
        'peak_detection': {'height': 0.15, 'prominence': 0.08, 'distance': 50}
    },
    'standard': {
        'name': 'Standard CNT Analysis',
        'description': 'Recommended for CNT characterization',
        'baseline': {'method': 'linear', 'n_points': 5},
        'normalization': 'minmax',
        'smoothing': {'window_length': 11, 'polyorder': 3},
        'peak_detection': {'height': 0.15, 'prominence': 0.08, 'distance': 50},
        'peak_fitting': {
            'd_band': {'n_peaks': 1, 'peak_type': 'lorentzian'},
            'g_band': {'n_peaks': 1, 'peak_type': 'lorentzian'},
            '2d_band': {'n_peaks': 1, 'peak_type': 'lorentzian'}
        }
    },
    'high_resolution': {
        'name': 'High Resolution',
        'description': 'Detailed analysis with fine peak detection',
        'baseline': {'method': 'als', 'lambda': 1e6, 'p': 0.01},
        'normalization': 'area',
        'smoothing': {'window_length': 7, 'polyorder': 3},
        'peak_detection': {'height': 0.05, 'prominence': 0.03, 'distance': 20}
    },
    'noisy_data': {
        'name': 'Noisy Data',
        'description': 'Optimized for noisy spectra',
        'baseline': {'method': 'als', 'lambda': 1e7, 'p': 0.05},
        'normalization': 'minmax',
        'smoothing': {'window_length': 21, 'polyorder': 3},
        'peak_detection': {'height': 0.2, 'prominence': 0.1, 'distance': 100}
    }
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================

UI_CONFIG = {
    'page_title': 'Raman Spectroscopy Analyzer',
    'page_icon': '📊',
    'layout': 'wide',
    'sidebar_state': 'expanded',
    'theme': {
        'primaryColor': '#1f77b4',
        'backgroundColor': '#ffffff',
        'secondaryBackgroundColor': '#f0f2f6',
        'textColor': '#262730'
    }
}

TAB_NAMES = {
    'overview': '📊 Data Overview',
    'preprocessing': '🔧 Preprocessing',
    'peak_detection': '🔍 Peak Detection',
    'peak_fitting': '📈 Peak Fitting',
    'analysis': '📋 Analysis & Results',
    'export': '💾 Export',
    'batch': '📦 Batch Processing',
    'comparison': '📊 Comparison'
}

# =============================================================================
# HELP TEXT AND TOOLTIPS
# =============================================================================

HELP_TEXT = {
    'baseline_linear': """
    **Linear Baseline Correction**
    
    Uses the first and last n points to fit a straight line, 
    which is then subtracted from the entire spectrum.
    
    Best for: Spectra with relatively flat baselines
    """,
    
    'baseline_als': """
    **Asymmetric Least Squares (ALS)**
    
    Advanced algorithm that fits a smooth baseline while 
    being asymmetric to handle peaks.
    
    Parameters:
    - Lambda: Controls smoothness (higher = smoother)
    - p: Asymmetry (lower = follows valleys more)
    
    Best for: Complex baselines with varying curvature
    """,
    
    'normalization_minmax': """
    **Min-Max Normalization**
    
    Scales all values to the range [0, 1]:
    I_norm = (I - I_min) / (I_max - I_min)
    
    Best for: Visual comparison and machine learning
    """,
    
    'peak_detection_height': """
    **Peak Height Threshold**
    
    Minimum intensity value for a point to be considered a peak.
    
    - Higher values: Only strong peaks detected
    - Lower values: More peaks detected (including noise)
    """,
    
    'peak_detection_prominence': """
    **Peak Prominence**
    
    Measures how much a peak stands out from its surroundings.
    
    - Higher values: Only distinct, well-separated peaks
    - Lower values: Shoulders and subtle features detected
    """,
    
    'peak_detection_distance': """
    **Minimum Distance**
    
    Minimum spacing between detected peaks (in data points).
    
    Prevents detecting multiple peaks on the same feature.
    """,
    
    'id_ig_ratio': """
    **ID/IG Ratio**
    
    Ratio of D-band to G-band intensities.
    
    Lower ratio = fewer defects = better quality
    
    Typical values:
    - < 0.8: High quality CNT
    - 0.8-1.2: Good quality
    - > 1.2: Lower quality or functionalized
    """,
    
    'crystallite_size': """
    **Crystallite Size (La)**
    
    Average size of ordered crystalline domains, 
    calculated using the Tuinstra-Koenig relation:
    
    La (nm) = C_λ / (ID/IG)
    
    Where C_λ depends on the laser wavelength.
    """
}

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    'file_not_found': 'File not found. Please check the file path.',
    'invalid_format': 'Invalid file format. Expected text file with two columns.',
    'insufficient_data': 'Insufficient data points for analysis.',
    'fitting_failed': 'Peak fitting failed. Try adjusting parameters or region.',
    'no_peaks_detected': 'No peaks detected. Try lowering the detection thresholds.',
    'baseline_failed': 'Baseline correction failed. Check your data.',
    'normalization_failed': 'Normalization failed. Check for invalid values.'
}

# =============================================================================
# DEVELOPER OPTIONS
# =============================================================================

DEBUG_MODE = False
VERBOSE_LOGGING = False
SHOW_WARNINGS = True

# Performance settings
CACHE_ENABLED = True
MAX_CACHE_SIZE_MB = 100
PARALLEL_PROCESSING = False  # For batch operations

# =============================================================================
# VERSION INFO
# =============================================================================

VERSION = '1.0.0'
RELEASE_DATE = '2024-01-01'
AUTHOR = 'Raman Analysis Team'
LICENSE = 'MIT'

# =============================================================================
# ADVANCED FEATURES (Future implementation)
# =============================================================================

ADVANCED_FEATURES = {
    'machine_learning': False,
    'cosmic_ray_removal': False,
    'background_fluorescence': False,
    'temperature_correction': False,
    'polarization_analysis': False
}
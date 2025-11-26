"""
Default settings and constants for Raman spectroscopy analysis
"""

# Default baseline correction parameters
BASELINE_DEFAULTS = {
    'linear_n_points': 5,
    'als_lambda': 1e6,
    'als_p': 0.01,
    'polynomial_degree': 3,
    'rolling_ball_window': 100
}

# Default normalization method
NORMALIZATION_DEFAULT = 'minmax'

# Default peak detection parameters
PEAK_DETECTION_DEFAULTS = {
    'height': 0.15,
    'prominence': 0.08,
    'distance': 50,
    'n_peaks': 11
}

# Default peak fitting parameters
PEAK_FITTING_DEFAULTS = {
    'd_band_range': (1200, 1450),
    'g_band_range': (1450, 1800),
    '2d_band_range': (2500, 3000),
    'peak_type': 'lorentzian',
    'max_iterations': 10000
}

# CNT characteristic bands
CNT_BANDS = {
    'D-band': {'range': (1200, 1450), 'typical': 1350, 'description': 'Defects/disorder'},
    'G-band': {'range': (1450, 1800), 'typical': 1580, 'description': 'Graphitic carbon'},
    '2D-band': {'range': (2500, 3000), 'typical': 2700, 'description': 'Second-order'}
}

# Plot styling
PLOT_CONFIG = {
    'figure_width': 14,
    'figure_height': 6,
    'line_width': 1.5,
    'marker_size': 10,
    'font_size': 12,
    'title_size': 14,
    'dpi': 100
}

# Export settings
EXPORT_FORMATS = ['CSV', 'Excel', 'JSON']
IMAGE_FORMATS = ['PNG', 'PDF', 'SVG']
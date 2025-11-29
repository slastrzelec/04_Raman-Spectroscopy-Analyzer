# 🔬 Raman Spectroscopy Analysis Tool

A comprehensive web application for processing, analyzing, and characterizing Raman spectroscopy data, with a focus on carbonaceous materials.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Overview

The Raman Spectroscopy Analysis Tool is built with Streamlit and advanced scientific Python libraries (Pandas, NumPy, SciPy, LMFIT) designed for chemists, materials scientists, and engineers to quickly and accurately process Raman spectroscopy data, particularly for carbonaceous materials like graphene and carbon nanotubes.

**Key Benefits:**
- Streamlines the entire workflow from raw data import to quantitative material property extraction
- Eliminates the need for complex desktop software
- Provides interactive, publication-ready visualizations
- Automates peak detection and fitting processes

## 📸 Screenshots

### Peak Detection Visualization
![Peak Detection](docs/images/peak_detection.png)
*Automatic peak identification with 11 detected peaks marked, showing configurable parameters like Minimum height (0.15) and Prominence (0.08)*

### Loaded Spectrum Overview
![Data Overview](docs/images/data_overview.png)
*Initial data validation screen displaying metadata: 7070 data points, wavenumber range (603.2 - 3400.0 cm⁻¹), and intensity statistics*

### Spectrum Normalization Comparison
![Preprocessing](docs/images/preprocessing.png)
*Side-by-side comparison of original (blue) and Min-Max normalized (green) spectra*

## 🛠️ Key Features

### 1. Data Handling
- Upload single or multiple `.txt` files with Wavenumber and Intensity columns
- Automatic data validation and metadata extraction
- Support for various file formats

### 2. Preprocessing
Essential steps to prepare spectra for analysis:
- **Baseline Correction**: Linear (Endpoints) and Asymmetric Least Squares (ALS) algorithms to remove fluorescence background
- **Normalization**: Min-Max, Max-Intensity, Area, and Vector (L2) normalization methods
- **Smoothing**: Savitzky-Golay filtering to reduce noise

### 3. Peak Analysis
- Automated detection and categorization of Raman bands (D, G, RBM, 2D bands)
- Calculation of fundamental peak statistics
- Interactive peak visualization

### 4. Quantitative Fitting & Deconvolution
- Advanced fitting using Lorentzian or Gaussian models
- Precise deconvolution of overlapping bands (e.g., D and G bands)
- Statistical quality metrics for fit assessment

### 5. Material Characterization
Calculate key material parameters:
- **I_D/I_G ratio**: Determines defect density and crystallinity
- **I_2D/I_G ratio**: Characterizes the number of graphene layers
- Additional material-specific metrics

### 6. Visualization & Export
- Interactive Plotly charts at every pipeline stage
- Export results in multiple formats
- Publication-ready figure generation

## 💻 Technical Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Frontend/App** | Streamlit | Rapid web application development and interactive UI/UX |
| **Data Processing** | Python (Pandas, NumPy) | Core data manipulation and scientific computing |
| **Signal Processing** | SciPy | Advanced algorithms (Savitzky-Golay, Peak Detection, ALS) |
| **Peak Fitting** | LMFIT | Non-linear least squares minimization for peak deconvolution |
| **Visualization** | Plotly / Matplotlib | Responsive, interactive data visualizations |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/slastrzelec/04_Raman-Spectroscopy-Analyzer.git
cd 04_Raman-Spectroscopy-Analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 📖 Usage

1. **Upload Data**: Click "Browse files" to upload your Raman spectroscopy `.txt` files
2. **Validate Data**: Review the data overview and metadata
3. **Preprocess**: Apply baseline correction, normalization, and smoothing as needed
4. **Detect Peaks**: Adjust detection parameters and identify Raman bands
5. **Fit Peaks**: Use Lorentzian/Gaussian models for precise deconvolution
6. **Analyze Results**: View calculated material properties and ratios
7. **Export**: Download processed data and visualizations

## 📁 Project Structure

```
04_Raman-Spectroscopy-Analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Sample data files
│   └── examples/
├── modules/               # Core functionality modules
│   ├── preprocessing.py   # Baseline correction, normalization
│   ├── peak_detection.py  # Peak finding algorithms
│   ├── fitting.py         # Peak fitting and deconvolution
│   └── visualization.py   # Plotting functions
└── docs/                  # Documentation and images
    └── images/
```

## 🔬 Scientific Background

### Raman Spectroscopy
Raman spectroscopy is a powerful analytical technique used to characterize molecular vibrations and crystal structures. For carbon materials, key bands include:

- **D band (~1350 cm⁻¹)**: Disorder-induced mode, indicates defects
- **G band (~1580 cm⁻¹)**: Graphitic mode, indicates sp² carbon
- **2D band (~2700 cm⁻¹)**: Overtone of D band, sensitive to layer number

### Material Characterization Metrics
- **I_D/I_G ratio**: Higher values indicate more defects and disorder
- **I_2D/I_G ratio**: Used to determine number of graphene layers (>2 for monolayer)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Stanisław Strzelec**
- GitHub: [@slastrzelec](https://github.com/slastrzelec)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Peak fitting powered by [LMFIT](https://lmfit.github.io/lmfit-py/)
- Scientific computing with [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/)

## 📚 References

For more information on Raman spectroscopy of carbon materials:
- Ferrari, A. C., & Robertson, J. (2000). Interpretation of Raman spectra of disordered and amorphous carbon. *Physical Review B*, 61(20), 14095.
- Malard, L. M., et al. (2009). Raman spectroscopy in graphene. *Physics Reports*, 473(5-6), 51-87.

---

Made with ❤️ for the materials science community
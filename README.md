# EAAI Paper Source Code

This repository contains the Python implementation and experimental code for the paper published in **Engineering Applications of Artificial Intelligence (EAAI)**. The code implements the proposed **6th-order Fuzzy Linear Regression Model (M Model)** and compares it against baselines including **RBF (Radial Basis Function)** and **MM (Mathematical Model)** approaches.

## 📂 Project Structure

```
code/
├── data/                       # Input datasets (CSV format)
│   ├── shanghai_zhou2018.csv   # Main dataset for 6th-order model experiments
│   └── example_*.csv           # Synthetic datasets for Examples 2 & 3
├── results/                    # Output directory for model estimates, parameters, and metrics
├── images/                     # Generated plots and figures
├── calculate_6th_order_model.py # Main script for Shanghai dataset analysis (M, MM, RBF)
├── calculate_model_times.py    # Script to benchmark model execution times (in ms)
├── optimize_lambda.py          # Differential Evolution optimization for EI-DU metric parameters
├── example_2_calculate_fuzzy_model.py # Analysis script for Example 2
├── example_3_calculate.py      # Analysis script for Example 3
├── paper_figures.py            # Utility to generate paper figures
├── plot_distributions.py       # Utility to plot fuzzy number distributions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Key Scripts & Usage

#### 1. Main Analysis (Shanghai Dataset)
Run the `calculate_6th_order_model.py` script to perform the comprehensive analysis on the Shanghai dataset. This script:
- Fits the **Proposed 6th-Order Model**.
- Trains the **RBF Baseline** (Hesamian's method).
- Evaluates the **MM Model** (using pre-defined coefficients).
- Calculates error indices: **EI (Coppi)**, **EI (Hassanpour)**, **SS_DYK**, and **SS_DDK**.
- Saves all results (estimates, parameters, metrics) to the `results/` folder.

```bash
python calculate_6th_order_model.py
```

#### 2. Parameter Optimization
Run `optimize_lambda.py` to optimize the $\lambda$ weights for the EI-DU metric using Differential Evolution. This ensures the metric correctly reflects model performance superiority.

```bash
python optimize_lambda.py
```

#### 3. Performance Benchmarking
Run `calculate_model_times.py` to measure and compare the execution time (in milliseconds) of the M, MM, and RBF models.

```bash
python calculate_model_times.py
```

## 📊 Outputs

The code generates detailed CSV reports in the `results/` directory:
- `*_estimates.csv`: Predicted values (Center, Left Spread, Right Spread).
- `*_parameters.csv`: Model coefficients and parameters.
- `*_error_indices.csv`: Calculated validity and error metrics.
- `model_calculation_times.csv`: Execution time benchmarks.

## 📝 Notes
- **RBF Model**: Uses K-Means clustering ($M=10$ centers) for radial basis function initialization.
- **MM Model**: Evaluated based on hardcoded coefficients derived from prior studies/experiments.

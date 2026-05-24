# Comparative Time Series Analysis and Forecasting of Mobile Network Traffic

This repository contains the implementation for a time series analysis and forecasting assignment on the **Telecom Italia Mobile (TIM) dataset** for the city of Milan, Italy.

---

## Project Overview

The dataset records mobile internet traffic across a 100×100 grid (10,000 geographical areas) of Milan over two months (November 2013 – January 2014). The project is divided into three tasks:

- **Task 1:** Efficient loading and memory optimization of a 5 GB dataset
- **Task 2:** Exploratory Data Analysis (EDA) — distribution, time series, stationarity, decomposition, ACF/PACF, spatial heatmap, anomaly detection
- **Task 3:** One-step-ahead forecasting using SARIMA, LSTM, and Transformer models, evaluated on December 16–22, 2013

---

## Repository Structure

```
├── milan_traffic_analysis.ipynb   # Main notebook (all tasks)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files excluded from version control
└── README.md                      # This file
```

> **Note:** The raw dataset files (.txt, .parquet) are not included in this repository due to their large size (~5 GB). See the Data Download section below for instructions.

---

## Requirements

- Python 3.11 or higher
- pip
- Windows, macOS, or Linux

---

## Setup Instructions

### Windows

```bash
# 1. Clone the repository
git clone https://github.com/falyseck/ML-MT1-Formative1/tree/main
cd ML-MT1-Formative1

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter in VS Code
# Open VS Code, then open milan_traffic_analysis.ipynb
# Select the venv kernel (top right corner)
```

### macOS / Linux

```bash
# 1. Clone the repository
git clone https://github.com/falyseck/ML-MT1-Formative1/tree/main
cd ML-MT1-Formative1

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook milan_traffic_analysis.ipynb
```

---

## Data Download

The dataset is not included in this repository. Download it from Harvard Dataverse:

1. **Telecom Activity Dataset (Milan):**
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV
   - Download all `.txt` files

2. **Grid Dataset (Milan):**
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU
   - Download the `.geojson` file

Once downloaded, place all files in the same folder and update the following paths at the top of the notebook (Cell 1 — Imports & Configuration):

```python
DATA_DIR   = r'C:\path\to\your\data\folder'
OUTPUT_PAR = r'C:\path\to\your\data\folder\traffic_optimized.parquet'
PLOTS_DIR  = r'C:\path\to\your\data\folder\plots'
GRID_FILE  = r'C:\path\to\your\data\folder\grid.geojson'
```

---

## How to Run

### Full Run (first time)

1. Open `milan_traffic_analysis.ipynb` in VS Code
2. Select your Python virtual environment as the kernel
3. Run cells sequentially from top to bottom using `Shift + Enter`
4. **Task 1** (data loading) takes approximately **10–15 minutes** on first run — it saves a `.parquet` file that speeds up all subsequent runs
5. **Task 3** (model training) takes approximately **60–90 minutes** on CPU

### Subsequent Runs (after parquet is saved)

- Task 1 loading cell automatically detects the existing `.parquet` file and skips the slow CSV loading — startup takes under 10 seconds
- You can run individual sections independently after the parquet file exists

### Running Only the Forecasting Models (Task 3)

If the parquet file already exists, you can skip directly to Task 3:

1. Run **Cell 1** (Imports & Configuration)
2. Run the **parquet loading cell** in Task 1 (last cell of Section 2)
3. Run the **helper functions cell** in Task 2
4. Run any Task 3 model cell directly

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 15 GB free |
| GPU | Not required | NVIDIA (speeds up Task 3) |

> **Note:** On 8 GB RAM, SARIMA with seasonal period 144 (10-minute resolution) will encounter a MemoryError during Kalman smoother allocation. The notebook handles this automatically by falling back to hourly resampling (period=24). This is documented and discussed in the report.

---

## Results Summary

| Model | Area 5059 MAE | Area 4159 MAE | Area 4556 MAE | Avg Train Time |
|---|---|---|---|---|
| SARIMA (hourly) | 85.20 | 14.55 | 16.84 | 3.6s |
| LSTM | **14.36** | **2.25** | **5.01** | 128.0s |
| Transformer | 20.14 | 3.13 | 7.04 | 537.4s |

**Best model: LSTM** — lowest MAE and RMSE across all three areas.

---

## References

- G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan," *Sci. Data*, 2015. https://doi.org/10.1038/sdata.2015.55
- S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, 1997.
- A. Vaswani et al., "Attention is all you need," *NeurIPS*, 2017.

---

## Video Demo

[https://youtu.be/UCv6H-sMXu8]

---


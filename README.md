# Early Cyber-Recon Detector

[![Python](https://img.shields.io/badge/Python->=3.10-blue.svg)](https://www.python.org/)
[![Engine](https://img.shields.io/badge/ETL-Polars-navy.svg)](https://pola.rs/)
[![Model](https://img.shields.io/badge/Classifier-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, high-performance intrusion detection pipeline focused on the **early detection of cyber reconnaissance and port scanning attacks**. 

By shifting from traditional, static flow-based features to a **continuous backward rolling window** (`df.rolling` via Polars), this project captures temporal sequence patterns without look-ahead bias, boosting the Precision-Recall Area Under Curve (**PR-AUC**) from **0.7498 to 0.9502** on the CIC-IDS2017 dataset.

---

## Performance Summary

Evaluation performed on **486,406 unseen test flows** (Stratified 80/20 train/test split, 6.53% positive class imbalance).

| Metric | Base Model (24 Features) | Correlated Rolling Model (27 Features) | Delta |
| :--- | :---: | :---: | :---: |
| **PR-AUC** | 0.7498 | **0.9502** | **+0.2004 (+26.7%)** |
| **ROC-AUC** | 0.9999 | **1.0000** | +0.0001 |
| **Precision (RECON)** | 0.99 | **1.00** | +0.01 |
| **Recall (RECON)** | 1.00 | **1.00** | 0.00 |

---

## Key Innovations

* **Continuous Backward Rolling Window:** Avoids step-function artifacts from discrete block windowing (`i // 50`). Computes fluid sequential aggregations across the previous 50 flows per row (`period="50i"`).
* **Leakage-Free Preprocessing:** Selective target filtering (BENIGN vs RECON) to avoid background attack contamination, and logarithmic scaling (log10(x + 1)) for heavy-tailed features.
* **Strict Scaler Isolation:** `RobustScaler` is fitted exclusively on the training set and applied to the test set in read-only mode.
* **Dynamic Class Weighting:** Automatic computation of LightGBM `scale_pos_weight` (~14.302) based on real-time training split ratios to mitigate class imbalance.
* **Informational Gain Supremacy:** The sequential feature `corr_avg_fwd_pkt_std_w50` ranks as the 2nd most impactful feature globally (11.6M gain points), proving the value of temporal context over isolated flow metrics.
* **Isolated Caching Architecture:** Thread-safe, independent disk caching for Base vs. Correlated datasets (`.npz`) and models (`.joblib`).

---

## Repository Structure

```text
cyber_recon_detector/
├── data/
│   ├── raw/                 # Raw dataset downloads (CIC-IDS2017)
│   └── processed/           # Compressed .parquet & cached .npz arrays
├── models/                  # Serialized .joblib models (base vs corr)
├── notebooks/
│   └── 03_model_evaluation.ipynb  # Evaluation notebook (2x2 grid & Gain analysis)
├── src/
│   └── cyber_recon_detector/
│       ├── features/        # preprocessor.py (df.rolling, scaling)
│       ├── models/          # trainer.py (LightGBM pipeline & metrics)
│       └── ingestion/       # dataset_loader.py & downloader.py
├── .gitignore
├── main.py                  # Pipeline execution entrypoint
└── README.md
```

--

## Quick Start

### 1. Prerequisites
* Python 3.10 or higher
* [Poetry](https://python-poetry.org/) (recommended) or standard `pip`

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/antsiri/Cyber-Reconnaissance-Detection.git
cd cyber-recon-detector

# Using Poetry
poetry install
```

### Running the Pipeline

To execute the complete ingestion, preprocessing, training, and evaluation pipeline:

```bash
# Execute via Poetry
poetry run python main.py

# Or via standard Python
python main.py
```

To force feature re-extraction or model retraining, adjust the configuration flags inside `main.py`:
 
```python
USE_CORRELATION = True
FORCE_RETRAIN = True
```
 
## Sequential Features Overview
 
The 3 engineered sequential correlation features added to the 24 base features (total 27) are:
 
- **`corr_unique_dst_ports_w50`**: Number of unique destination ports targeted within the backward 50-flow window. Instantly identifies horizontal and vertical port sweeps.
- **`corr_avg_fwd_pkt_std_w50`**: Average forward packet length standard deviation across the window. Detects the rigid structural behavior and monotony of automated probing tools (e.g., Nmap, Masscan) compared to variable benign traffic.
- **`corr_avg_duration_w50`**: Mean flow duration over the window. Flags rapid-fire, short-lived probing connections typical of TCP SYN scans.
## Model Evaluation & Notebooks
 
To reproduce the performance visualizer (which opens the 2x2 Precision-Recall / ROC grids and feature importance bar charts), run the evaluation notebook:
 
```bash
jupyter notebook notebooks/03_model_evaluation.ipynb
```
 
## Top Features by Information Gain
 
The Gain metric measures the actual reduction in entropy (loss) brought by each feature across all LightGBM trees.
 
| Rank | Category | Feature Name | Total Gain | Split Count |
|------|----------|---------------|------------|-------------|
| 1 | [BASE] | Packet Length Mean | 20,533,472.70 | 190 |
| 2 | [CORR] | corr_avg_fwd_pkt_std_w50 | 11,616,661.49 | 439 |
| 3 | [BASE] | Bwd Packet Length Mean | 4,071,736.04 | 116 |
| 4 | [BASE] | Fwd IAT Mean | 1,734,072.83 | 131 |
| 14 | [CORR] | corr_avg_duration_w50 | 48,510.81 | 318 |
 
This confirms that despite requiring fewer atomic splits than some noisy continuous features, the sequential rolling window metrics hold massive predictive power, allowing the trees to eliminate uncertainty with high precision.
 
## License
 
Distributed under the MIT License. See `LICENSE` for more information.

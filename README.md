# Explainable Financial Fraud Detection using Graph Neural Networks

An end-to-end educational project that analyses PaySim synthetic mobile-money transactions, compares transaction-level Random Forest baselines with graph neural network (GCN/GAT) edge classifiers, explains Random Forest predictions with SHAP, and provides an interactive Streamlit dashboard for fraud-risk investigation.

## Project components

- **Notebook:** `01_fraud_gnn_pipeline.ipynb` — data quality checks, exploratory analysis, feature engineering, graph construction, chronological train/validation/test split, model training/evaluation, SHAP explanations, and artifact generation.
- **Shared features:** `fraud_features.py` — feature engineering and SHAP helper functions shared by the notebook and dashboard.
- **Dashboard:** `app.py` — three views: Risk overview, Investigate alerts, and Score a transaction.
- **Artifacts:** generated locally in `saved_models/` by notebook Section 12.1.

## Dataset

The project uses PaySim, a **synthetic** mobile-money transaction dataset. The notebook expects `PS_20174392719_1491204439457_log.csv`. See [`data/README.md`](data/README.md) for the path detail.

Do not commit the raw CSV or large generated model files to GitHub.

## Installation

Use Python 3.10 or 3.11 if possible, and a virtual environment. Install PyTorch and PyTorch Geometric according to the official installation instructions for your Python and CPU/CUDA setup; their compatible versions can vary.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a working environment, you do not need to reinstall everything just to publish the repository.

## Run the notebook

Open `01_fraud_gnn_pipeline.ipynb` in Jupyter. The notebook uses `DATA_PATH = Path("../data/PS_20174392719_1491204439457_log.csv")`; ensure the CSV is at that relative location or update the path for your machine.

The notebook can be computationally intensive. It is not necessary to rerun model training merely to start the dashboard if the saved artifacts already exist.

## Run the dashboard

After the notebook's Section 12.1 has generated the expected artifacts, run this from the directory containing `app.py`:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (typically `http://localhost:8501`).

The dashboard expects these files in `saved_models/`:
- `fraud_rf.joblib`
- `dashboard_scored.csv.gz`
- `model_comparison.csv`

## Evaluation notes and limitations

- PaySim is simulated data; high scores do not imply real-world performance.
- The project compares all-feature and pre-transaction-only Random Forest models to examine reliance on post-transaction balances.
- Thresholds are selected on validation data and applied to the held-out test period.
- Fraud is highly imbalanced, so PR-AUC, precision, recall, F1, confusion matrices, and top-k alert quality are more informative than accuracy alone.
- Account relationships in PaySim can be sparse, limiting the benefits of message passing.

## Repository hygiene

The `.gitignore` excludes raw data, generated model bundles, scored CSVs, virtual environments, and notebook checkpoints. Keep only source code, documentation, and small non-sensitive figures in the public repository.

## Responsible use

This is an educational portfolio project, not a production fraud-detection system. Do not use its predictions for real financial decisions without independent validation, privacy/security review, and human oversight.

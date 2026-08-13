# Predictive Modeling Optimization Challenge (IIT KGP ML Hackathon)

**Team Name:** AlgoRythm  
**Submission File:** [AlgoRythm.csv](file:///c:/Users/prith/Projects/IIT_KGP_ML/AlgoRythm.csv)

This repository contains the machine learning surrogate model developed by **Team AlgoRythm** to predict the **Overall Yield of Product B** in a non-isothermal continuous flow reactor.

---

## 📌 Project Overview

The continuous flow reactor operates with a competing series-parallel reaction network:
1. **Desired Reaction**: $A \xrightarrow{k_1} B$
2. **Side Reaction**: $B \xrightarrow{k_2} C$

This model serves as a computational surrogate to instantaneously predict the reactor exit yield (`overall_yield`) across varying operating parameters (flow rate, inlet concentration, thermal profiles, and reactor length).

---

## 🤖 Why PyCaret Was Chosen & Its Key Advantages

**PyCaret** is an open-source, low-code machine learning library in Python that automates machine learning workflows. For this continuous flow reactor optimization challenge, PyCaret provided several decisive technical advantages:

### Key Advantages for Chemical Reactor Modeling:

1. **Rapid Automated Model Comparison (`compare_models()`)**:
   * Evaluates 10+ regression algorithms (e.g., LightGBM, Extra Trees, Random Forest, XGBoost, Ridge Regression) simultaneously under unified 10-fold cross-validation.
   * Enables immediate ranking by **RMSE** (the hackathon's primary evaluation metric), quickly identifying algorithms that capture non-linear chemical dynamics.

2. **Automated Feature Preprocessing & Scaling**:
   * Standardizes continuous physical variables (temperature, concentration, flow rates) via internal pipeline transformations, preventing scale imbalance between parameters.

3. **Overfitting Prevention & Cross-Validation**:
   * Built-in stratified/k-fold cross-validation guarantees robust local evaluation on small industrial datasets (150 training samples), preventing data leakage.

4. **Seamless Model Finalization & Serialization**:
   * `finalize_model()` retrains the top-performing estimator on the complete dataset before export via `save_model()`, producing high-fidelity inference pipelines.

---

## 💡 How to Use PyCaret in This Project

PyCaret simplifies end-to-end model building into a few clean Python calls:

### 1. Initialize Experiment Setup (`setup`)
```python
from pycaret.regression import setup

reg_setup = setup(
    data=train_df,
    target='overall_yield',
    session_id=123,
    train_size=0.8,
    normalize=True,
    fold=10
)
```

### 2. Compare Models & Select the Best Estimator (`compare_models`)
```python
from pycaret.regression import compare_models

# Trains and evaluates 10+ regressors, returning the top model based on RMSE
best_model = compare_models(sort='RMSE')
```

### 3. Finalize & Save Model Pipeline (`finalize_model` & `save_model`)
```python
from pycaret.regression import finalize_model, save_model

# Retrain best model on the complete dataset (train + validation)
final_model = finalize_model(best_model)

# Save pipeline as a binary pickle artifact (.pkl)
save_model(final_model, 'best_yield_model')
```

### 4. Batch Predict Unseen Test Data (`predict_model`)
```python
from pycaret.regression import predict_model

# Generate predictions on unseen test dataset
predictions = predict_model(final_model, data=test_df)
# Predictions are stored in the 'prediction_label' column
```

---

## 🛠️ Environment Prerequisites

- **Python**: `3.11.x` *(Recommended: Python 3.11 for pre-compiled PyCaret binary wheel compatibility)*
- **OS**: Windows, macOS, or Linux

---

## 🚀 Setup & Installation Guide

Follow these step-by-step instructions to run this project on any machine.

### 1. Clone the Repository
```bash
git clone <repository-url>
cd IIT_KGP_ML
```

### 2. Create and Activate a Virtual Environment

#### On Windows (PowerShell / Command Prompt):
```powershell
# Using Py Launcher for Python 3.11
py -3.11 -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1
# (or in CMD: .\.venv\Scripts\activate.bat)
```

#### On macOS / Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade Core Build Tools & Install Dependencies
```bash
python -m pip install --upgrade pip setuptools wheel
pip install pandas pycaret ipykernel notebook
```

### 4. Register Environment with Jupyter Kernel
```bash
python -m ipykernel install --user --name=kgp_ml_env --display-name "Python 3.11 (KGP ML)"
```

---

## 📊 Directory Structure

```text
IIT_KGP_ML/
├── datasets/
│   ├── train_dataset.csv     # Training data (150 rows)
│   └── test_dataset.csv      # Unseen test data (50 rows)
├── assets/
│   └── IIT_KGP_ML.pdf        # Problem statement document
├── Model.ipynb               # End-to-end model training & inference notebook
├── AlgoRythm.csv             # Final Hackathon Submission File (50 rows, 1 column: overall_yield)
├── README.md                 # Setup, PyCaret guide & execution details
└── .gitignore                # Git ignore rules
```

---

## 🏃 Running the Notebook

1. Launch Jupyter Notebook or open VS Code:
   ```bash
   jupyter notebook
   ```
2. Open `Model.ipynb`.
3. Select kernel: **`Python 3.11 (KGP ML)`** (or `.venv`).
4. Run all cells sequentially to train the PyCaret regression model and output predictions for `test_dataset.csv`.

---

## 🎯 Generating Submission File (`AlgoRythm.csv`)

To format predictions for the hackathon submission:

```python
import pandas as pd
from pycaret.regression import load_model, predict_model

# Load model and test dataset
model = load_model('best_yield_model')
test_df = pd.read_csv('datasets/test_dataset.csv')

# Generate predictions
predictions = predict_model(model, data=test_df)

# Create submission dataframe matching hackathon criteria
submission = pd.DataFrame({
    'overall_yield': predictions['prediction_label'].round(3)
})

# Save to team CSV file
submission.to_csv('AlgoRythm.csv', index=False)
```

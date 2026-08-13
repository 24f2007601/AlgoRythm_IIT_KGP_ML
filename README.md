# Predictive Modeling Optimization Challenge (IIT KGP ML Hackathon)

This repository contains the machine learning surrogate model developed to predict the **Overall Yield of Product B** in a non-isothermal continuous flow reactor.

---

## 📌 Project Overview

The continuous flow reactor operates with a competing series-parallel reaction network:
1. **Desired Reaction**: $A \xrightarrow{k_1} B$
2. **Side Reaction**: $B \xrightarrow{k_2} C$

This model serves as a computational surrogate to instantaneously predict the reactor exit yield (`overall_yield`) across varying operating parameters (flow rate, inlet concentration, thermal profiles, and reactor length).

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
├── README.md                 # Setup and execution guide
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

## 🎯 Generating Submission File

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

# Save to your team CSV file
submission.to_csv('YourTeamName.csv', index=False)
```

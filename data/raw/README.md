# Dataset Documentation

## Primary Dataset: Credit Card Fraud Detection
The canonical dataset for this project is the **Credit Card Fraud Detection Dataset**, originally compiled by the Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB).

### Overview
* **Total Transactions:** 284,807
* **Fraudulent Transactions:** 492 (0.172% fraud rate — highly imbalanced)
* **Timeframe:** 2 days of transactions in September 2013 by European cardholders
* **Features:** 31 numerical columns:
  * `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
  * `V1` – `V28`: Principal components obtained via PCA (anonymized for user confidentiality).
  * `Amount`: Transaction amount in Euros/Currency.
  * `Class`: Ground-truth label (`0` = Genuine, `1` = Fraudulent).

---

## How to Obtain the Dataset

### Option 1: Automated Download via Project CLI (Recommended)
You can automatically download the official dataset into `data/raw/creditcard.csv` using:
```bash
python run.py --download-data
```
or via the Python module:
```bash
python -c "from src.data_loader import download_creditcard_dataset; download_creditcard_dataset()"
```

### Option 2: Download from Kaggle
1. Visit: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv` (approx. 150 MB).
3. Place the file at:
   ```text
   data/raw/creditcard.csv
   ```

### Option 3: Fast Benchmark / Offline Generation
If you want to train, test, or verify the pipeline quickly without downloading 150 MB:
```bash
python run.py --generate-sample
```
This creates a statistically realistic synthetic benchmark with the exact 31-column schema, preserving extreme class imbalance, power-law amounts, and multivariate PCA distributions.

---

## Important Data Handling Rules
* Never commit raw or processed CSV files to Git (enforced via `.gitignore`).
* All preprocessing scalers must be fitted **exclusively on the training split** to prevent data leakage.
* Holdout test data must remain strictly untouched until final model evaluation.

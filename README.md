# Energy Prediction with LSTM & Transformer

This repository implements a time series forecasting pipeline for predicting daily **global active power consumption** using deep learning models including **LSTM** and **Transformer**. It is designed for clean, structured power usage data with temporal and numeric features.

---

## 📦 Environment

- Python 3.8+
- PyTorch 1.11.0
- pandas
- scikit-learn
- matplotlib
- numpy

> ⚠️ You can set up the environment using:
> ```bash
> conda create -n power_pred python=3.8
> conda activate power_pred
> pip install -r requirements.txt
> ```

---

## 📊 Dataset

The dataset contains power usage logs per minute and other related features. The key field to predict is:

- `Global_active_power`: total active power consumption (in kilowatt)

Data is preprocessed into daily aggregates using:
- Sum of power-related fields
- Mean of voltage and intensity
- First values of supplementary features (RR, NBJRR1, etc.)
- Removal of rows with missing values or invalid entries (`?`, empty cells)

Cleaned CSV files:
- `dataset/train_cleaned.csv`
- `dataset/test_cleaned.csv`

---

## 🧪 Preprocessing

Data preprocessing is performed in `data/dataprocess.py`:
- Parse `DateTime`
- Group by date
- Aggregate features
- Generate sequences for model input
- Normalize input features

You can run preprocessing via:

```bash
python preprocess.py

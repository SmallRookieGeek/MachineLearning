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

You can run preprocessing via:

```bash
python preprocess.py

## 🧠 Models

You can choose the model architecture via the `--model` argument when running the script:

- `--model LSTM`: use LSTM-based sequential model
- `--model Transformer`: use Transformer with attention mechanism
- `--model Logical`: use logic-aware numerical embedding model (e.g., with box or sinusoidal encodings)

Example:

```bash
python run2.py --model LSTM
python run2.py --model Transformer
python run2.py --model Logical



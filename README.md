# Universal Anomaly Detector

A full-stack machine learning application for detecting anomalous records in tabular datasets using Autoencoders.

## Overview

Universal Anomaly Detector is designed to work with different tabular datasets rather than being limited to a single dataset.

Users can upload CSV or Excel files, preprocess the data, train an Autoencoder model, and identify unusual records using reconstruction error.

## Features

- Upload CSV and Excel datasets
- Automatic dataset preprocessing
- Numerical and categorical feature handling
- Automatic feature scaling
- Dynamic Autoencoder architecture
- Reconstruction-error-based anomaly detection
- Adaptive anomaly threshold
- Large dataset support
- Anomaly statistics and results
- Optional evaluation using ground-truth labels
- Clean web-based interface

## Technology Stack

### Frontend
- React
- Vite
- TypeScript

### Backend
- Python
- FastAPI
- Uvicorn

### Machine Learning
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy

## Project Structure

```
Universal-Anomaly-Detector
├── backend
│   ├── app
│   └── requirements.txt
├── frontend
│   ├── src
│   ├── package.json
│   └── vite.config.ts
├── sample_data.csv
├── README.md
└── .gitignore
```

## How It Works

```
Dataset Upload
      ↓
Data Preprocessing
      ↓
Feature Transformation
      ↓
Autoencoder Training
      ↓
Reconstruction Error
      ↓
Adaptive Threshold
      ↓
Normal / Anomaly Classification
      ↓
Results
```

The Autoencoder learns patterns in the dataset. Records with significantly higher reconstruction errors are treated as potential anomalies.

## Running Locally

### Backend

```bash
cd backend
py -3.11 -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

API documentation:

`http://127.0.0.1:8000/docs`

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

`http://localhost:5173`

## Dataset Support

The application is designed for tabular datasets in CSV, XLSX, and XLS formats. It can handle numerical and categorical features.

If a dataset contains a ground-truth anomaly or classification label, it can be used for evaluation.

## Example Use Cases

- Fraud detection
- Network anomaly detection
- Transaction monitoring
- Sensor data analysis
- System monitoring
- Operational data analysis

## Machine Learning Approach

The system uses an Autoencoder-based unsupervised anomaly detection approach. The model attempts to reconstruct input features, and reconstruction error is calculated for each record.

A higher reconstruction error indicates that a record differs significantly from patterns learned by the model and may therefore be anomalous.

## Important

Large datasets should not be committed to the repository. Datasets such as `creditcard.csv` should remain local and be excluded through `.gitignore`.

## Author

**Vijaya Shree**

Computer Science Engineering Student

GitHub: https://github.com/VijayaShree-23

# Universal Anomaly Detector

Clean full-stack anomaly detection for arbitrary tabular CSV/XLSX/XLS datasets.

## Features
- Automatic schema analysis and preprocessing
- Numerical + categorical feature handling
- Automatic ID/constant-column filtering
- Automatic label detection (`Class`, `Fraud`, `Anomaly`, `Label`, etc.)
- Dynamic autoencoder model
- Large-dataset handling with sampled training and full-dataset scoring
- Adaptive reconstruction-error threshold
- Precision, Recall and F1 when ground-truth labels are available
- Clean professional dashboard

## Run backend

Use **Python 3.11.x**.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

## Run frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Supported datasets
CSV, XLSX and XLS tabular files. Large files are accepted; the model samples training rows but scores the complete dataset.

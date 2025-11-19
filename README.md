# IDS Streamlit App

This repository contains a Streamlit app to explore and train models for the IDS project.

Files added:

- `app.py` — Streamlit application. Load the training and test CSV files (or upload your own), preprocess, select features (RFE), train a model and download the trained model as a pickle.
- `requirements.txt` — Python packages needed to run the app.

How to run locally (Windows, using the workspace's virtual environment):

1. Activate your virtual environment (if you used the workspace `.venv`):

```powershell
C:/Users/omsco/OneDrive/Desktop/IDS/.venv/Scripts/Activate.ps1
```

2. Install requirements:

```powershell
python -m pip install -r requirements.txt
```

3. Run Streamlit:

```powershell
streamlit run simulation_app.py
```

The app expects `Train_data.csv` and `Test_data.csv` in the repository root by default. You can instead upload your own CSVs via the sidebar.

Deployment options:

- Streamlit Community Cloud: Create a new app and link this repository; specify `streamlit run app.py` as the start command.
- Other: Dockerize or deploy on any cloud provider that supports Python web apps.

Notes and limitations:

- The app runs feature selection (RFE) using a small RandomForest; for large datasets this will be slower.
- The app trains models in-process — for production use, pretrain and serve a model or use background jobs.


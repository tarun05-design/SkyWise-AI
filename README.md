# 🎫 SkyWise AI — Flight Delay Predictor (Boarding Pass Edition)

[![Streamlit App](https://static.streamlit.io/badge_github.svg)](https://skywise-ai.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🌐 **Live Demo:** [https://skywise-ai.streamlit.app/](https://skywise-ai.streamlit.app/)

**SkyWise AI** is a premium Streamlit web application designed to forecast whether a flight will experience an arrival delay of **15 minutes or more**. 

Styled with a dark-mode glassmorphic interface inspired by airline boarding passes, SkyWise AI combines machine learning predictions with subtle, satisfying micro-animations to deliver an authentic ticket-checking experience.

---

## ✨ Key Features

* **Dual-Model Support**: Compare predictions from **XGBoost** (optimized for raw accuracy) and **LightGBM** (optimized for leaf-wise speed) side-by-side.
* **Authentic Ticket Styling**: The prediction output dynamically generates a boarding pass complete with flight paths, scheduled departure/arrival grids, and an animated delay stamp.
* **Instant Presets**: Launch predictions in one click with pre-loaded, realistic itineraries (e.g., *ATL → FLL*, *JFK → LAX*, *ORD → DEN*).
* **Airport Database**: Auto-fills locations and state details for **369 US airports** using an embedded lookup database.
* **Advanced Options**: Supports overrides for operating airlines and codeshare partners.

---

## 🛠️ Technology Stack & Requirements

The project uses the following libraries:
* **UI Framework**: [Streamlit](https://streamlit.io/) (v1.32+)
* **Data Processing**: [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
* **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/) and [LightGBM](https://lightgbm.readthedocs.io/)
* **Pre-processing**: [Scikit-Learn](https://scikit-learn.org/) (for fitted categorical label encoders)

---

## 📁 Repository Structure

```tree
Flight Delay Prediction/
├── app.py                     # Streamlit application source code & styling
├── requirements.txt           # Project python dependencies
├── .gitignore                 # Configured git ignore file
├── Data/
│   ├── cleaned_flight_delay_data.csv  # Dataset used for evaluation
│   └── sample.xlsx            # Sample spreadsheet data
├── Models/
│   ├── flight_delay_xgboost_model.pkl # Pre-trained XGBoost classifier
│   └── flight_delay_lgbm_model.pkl    # Pre-trained LightGBM classifier
└── Utils/
    └── label_encoders.pkl     # Categorical label encoders
```

---

## 🚀 Local Installation & Quick Start

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/tarun05-design/SkyWise-AI.git
cd SkyWise-AI
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Make sure you install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Start the Streamlit server:
```bash
streamlit run app.py
```
Your default web browser should open automatically and load the application at `http://localhost:8501`.

---

## 📊 How It Works (Machine Learning)

1. **Feature Engineering**: Derives cyclical calendar features (Quarter, Month, Day of Month, Day of Week) from the flight date. Scheduled departure and arrival times are mapped to standard department time blocks (e.g., `1600-1659`).
2. **Label Encoding**: Uses pre-fitted label encoders stored in `Utils/label_encoders.pkl` to convert high-cardinality categories (Airports, Airlines) into model-ready integers.
3. **Prediction Engine**: Runs the features through the selected Gradient Boosted Classifier (XGBoost or LightGBM) to compute:
   - Binary delay outcome (On-Time vs. Delayed $\ge 15$ mins)
   - Probability score for both outcomes (rendered as dynamic progress bars).

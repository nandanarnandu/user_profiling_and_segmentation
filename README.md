# 🧠📊 AI-Powered User Profiling & Segmentation

[![Python](https://img.shields.io/badge/python-v3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-v2.3%2B-black.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A complete Flask-based machine learning app that profiles and segments users using demographics, behavior, and interests to power targeted advertising strategies. Built with Python, Flask, Pandas, and Scikit-learn.

---

## ✨ Features

- 🔐 Upload & Manage Data  
  Import CSVs with demographic, behavioral, and interest features

- 🔎 EDA at a Glance  
  Summary stats, missing value report, distributions, and correlations

- 🧼 Preprocessing Pipeline  
  Scaling (Standard/MinMax), one-hot encoding, and feature selection

- 🧩 Clustering (K-Means)  
  Configurable K, inertia/Elbow and Silhouette diagnostics

- 🗺️ Segment Insights & Labels  
  Auto-generated segment summaries (e.g., “Weekend Warriors”, “Engaged Professionals”, “Budget Browsers”)

- 📊 Visual Analytics  
  Radar charts for segment profiles, cluster counts, PCA 2D plot

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/user-profiling-segmentation.git
cd user-profiling-segmentation

python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# (Optional) set environment variables
# Linux/Mac:
export FLASK_APP=app.py
export FLASK_ENV=development
# Windows (PowerShell):
# $env:FLASK_APP="app.py"
# $env:FLASK_ENV="development"

# Run the app
flask run

# Open http://127.0.0.1:5000

```


---

## 📂 Dataset

The dataset includes demographic, behavioral, and interest-based attributes for ad users.
Typical Columns
⦁	**Demographics**: age, gender, income_level

⦁	**Device & Usage**: device_type, time_spent_weekday, time_spent_weekend

⦁	**Engagement**: likes, reactions, ctr (click-through-rate)

⦁	**Interests**: top_interests (or one-hot encoded interest columns)

⦁	Place your CSV inside data/ (e.g., data/ad_users.csv) or upload via the web UI.

## 📊 Features Used

⦁	Age, Gender, Income Level

⦁	Device Usage

⦁	Time Spent Online (Weekday & Weekend)

⦁	Likes and Reactions

⦁	Click-Through Rate (CTR)

⦁	Top Interests

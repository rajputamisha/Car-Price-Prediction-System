# 🚗 Car Price Prediction System

A Machine Learning-based web application that predicts the resale price of used cars based on various attributes like make, model, year, mileage, and fuel type. Built using Python, Scikit-learn, Flask, and a user-friendly web interface.

---

## 📌 Project Overview

Estimating the resale value of a car is often complex due to factors like brand, mileage, fuel type, and condition. This project uses machine learning algorithms—Random Forest, Gradient Boosting, and Linear Regression—to provide accurate and consistent car price predictions. The aim is to support buyers, sellers, and dealers with a transparent, automated tool for car valuation.

---

## 🎯 Features

- Input parameters: Car brand, model, year, mileage, fuel type
- Predicts current and future resale value (e.g., after 2 years)
- User-friendly GUI using HTML, CSS, JavaScript
- Backend built with Flask for real-time predictions
- Trained on cleaned and preprocessed historical data
- Stores predictions in MySQL for tracking and future use

---

## 🧠 Machine Learning Models Used

- *Linear Regression:* Baseline model; easy to interpret but limited with non-linear data.
- *Random Forest:* Ensemble method using decision trees; reduces overfitting and improves accuracy.
- *Gradient Boosting:* Combines weak learners for strong predictive power on complex patterns.

---

## 🧪 Evaluation Metrics

- *RMSE (Root Mean Squared Error):* 4.5%
- *MAE (Mean Absolute Error):* Low error rates across all models
- *R² Score:* 0.92 (Random Forest & Gradient Boosting)

---

## 🛠 Technologies Used

| Layer        | Tools/Technologies                 |
|--------------|------------------------------------|
| Programming  | Python 3.10+, Jupyter Notebook     |
| Libraries    | Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn |
| Web Backend  | Flask                              |
| Web Frontend | HTML, CSS, JavaScript              |
| Database     | MySQL                              |

---

## 📊 System Design

- *Frontend:* Collects car details from the user via a responsive web form.
- *Backend:* Receives inputs via REST API, processes them using trained ML models, and returns predicted prices.
- *Database:* Stores historical data and prediction logs for reference and analysis.

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction

# 📊 Vendor Invoice Analytics & ML Prediction System

An end-to-end machine learning project for analyzing vendor invoice data and supporting procurement decision-making through:

1. **Invoice Risk Detection** — classifying invoices as normal or potentially risky.
2. **Freight Cost Prediction** — predicting freight costs using vendor invoice information.
3. **Interactive Streamlit Web Application** — providing a user-friendly interface for real-time predictions.

---

## 🚀 Project Overview

Vendor invoice data contains valuable information about purchasing, invoice amounts, freight charges, receiving delays, and payment timelines. This project uses machine learning to extract insights from this data and automate two important tasks:

### 🔴 Invoice Risk Detection

The system identifies potentially risky invoices using business rules and machine learning.

An invoice is flagged when:

* The invoice amount differs significantly from the total purchase item value.
* The receiving delay is abnormally high.

The generated target variable is:

```text
flag_invoice

0 → Normal Invoice
1 → Potentially Risky Invoice
```

Several classification models were explored, including:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The Random Forest model was further optimized using `GridSearchCV` and F1-score optimization.

---

### 🚚 Freight Cost Prediction

The regression component predicts freight cost based on vendor invoice information.

The initial model uses:

```text
Dollars, Quantity → Freight
```

Multiple regression algorithms were compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

Models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The model with the lowest MAE is selected as the final freight prediction model.

---

# 🧠 Machine Learning Workflow

```text
                 ┌─────────────────────┐
                 │   Raw CSV Dataset   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   MySQL Database    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Extraction &   │
                 │ Feature Engineering │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
   ┌─────────────────────┐     ┌─────────────────────┐
   │ Invoice Risk        │     │ Freight Cost        │
   │ Classification      │     │ Regression          │
   └──────────┬──────────┘     └──────────┬──────────┘
              │                           │
              ▼                           ▼
   ┌─────────────────────┐     ┌─────────────────────┐
   │ Random Forest       │     │ Model Comparison    │
   │ Classifier          │     │ & Selection         │
   └──────────┬──────────┘     └──────────┬──────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Saved ML Models     │
                 │ (.pkl files)        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Web App   │
                 └─────────────────────┘
```

---

# 📂 Project Structure

```text
vendor-invoice-analysis/
│
├── app.py
│
├── freight cost prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
│
├── invoice flagging/
│   ├── data_preprocessing.py  
│   ├── model_evaluation.py  
│   └── train.py
│
├──inferencing/ 
│   ├── predict_invoice_flag.py
│   └── predict_freight.py
│
├── notebooks/
│   ├── invoice flagging.ipynb
│   └── predicting freight cost.ipynb
│
├── models/
│   ├── predict_flag_invoice.pkl
│   ├── predict_freight_model2.pkl
│   └── scaler.pkl
│
├── requirements.txt
│
└── README.md
```

---

# 🗄️ Data Pipeline

The project uses a MySQL database for storing and querying vendor and purchase information.

The invoice risk pipeline combines:

### Purchase Information

* Purchase Order Number
* Brand
* Quantity
* Dollars
* Purchase Order Date
* Receiving Date

### Vendor Invoice Information

* Invoice Quantity
* Invoice Dollars
* Freight
* Invoice Date
* Pay Date

The project creates aggregated purchase-level features including:

```text
total_brands
total_item_quantity
total_item_dollars
avg_receiving_delay
```

Additional invoice-level features include:

```text
invoice_quantity
invoice_dollars
Freight
days_PO_to_invoice
days_to_pay
```

---

# 🔴 Invoice Risk Detection

## Target Creation

A business-rule-based label is created:

```python
def create_invoice_risk_label(row):

    if abs(
        row["invoice_dollars"]
        - row["total_item_dollars"]
    ) > 5:
        return 1

    if row["avg_receiving_delay"] > 10:
        return 1

    return 0
```

### Interpretation

| Label | Meaning                   |
| ----- | ------------------------- |
| `0`   | Normal Invoice            |
| `1`   | Potentially Risky Invoice |

---

## Classification Features

The classification model uses features such as:

```text
total_brands
total_item_quantity
total_item_dollars
invoice_quantity
invoice_dollars
Freight
days_PO_to_invoice
```

---

## Models Compared

### 1. Linear Regression

Provides a simple baseline relationship between invoice dollar value and freight cost.

### 2. Decision Tree Regressor

Captures non-linear relationships.

### 3. Random Forest Regressor

Uses an ensemble of decision trees to improve predictive performance.

---

## Evaluation Metrics

### Mean Absolute Error (MAE)

Measures the average absolute prediction error.

```text
Lower MAE = Better
```

### Root Mean Squared Error (RMSE)

Penalizes larger errors more heavily.

```text
Lower RMSE = Better
```

### R² Score

Measures how much variation in freight cost is explained by the model.

```text
Higher R² = Better
```

The final model is selected based on the lowest MAE.

---

# 🌐 Streamlit Web Application

The project includes an interactive Streamlit application that allows users to make predictions without directly writing Python code.

The application provides:

## 🔴 Invoice Risk Detection

Users enter invoice and purchase information such as:

```text
Total Item Quantity
Total Item Dollars
Invoice Quantity
Invoice Dollars
Freight
```

The application returns:

```text
Normal Invoice
```

or:

```text
Potentially Risky Invoice
```

The application can also display the predicted risk probability.

---

## 🚚 Freight Cost Prediction

Users enter relevant invoice information and the trained regression model returns the predicted  freight cost associated with a vendor invoice..

Example:

```text
Input:
Invoice Dollars, Invoice Quantity

Output:
Predicted Freight Cost
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Database

* MySQL
* SQLAlchemy
* PyMySQL

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Development Environment

* Jupyter Notebook
* VS Code

---

# 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
sqlalchemy
pymysql
streamlit
```

---

# ▶️ Running the Streamlit Application

From the project root directory:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

# 🧪 Running the Machine Learning Pipelines

The notebooks used for development are:

```text
invoice flagging.ipynb
predicting freight cost.ipynb
```

The notebooks contain:

* Data loading
* MySQL integration
* Exploratory Data Analysis
* Feature engineering
* Statistical testing
* Model training
* Model evaluation
* Hyperparameter tuning
* Prediction experiments

---

# 📊 Exploratory Data Analysis

The project includes:

* Descriptive statistics
* Missing-value analysis
* Correlation analysis
* Feature relationship analysis
* Distribution analysis
* Invoice risk class distribution
* Freight cost analysis

A statistical t-test was also used to compare flagged and normal invoice groups and identify statistically significant differences between groups.

---

# 💾 Saved Models

The trained models are stored in the `models/` directory:

```text
predict_flag_invoice.pkl
```

Used for:

```text
Invoice Risk Classification
```

```text
predict_freight_model.pkl
```

Used for:

```text
Freight Cost Prediction
```

```text
scaler.pkl
```

Used to ensure that prediction data receives the same preprocessing as the training data.

---

# 🎥 Application Demo

A demonstration video of the Streamlit application is included with the project.

The demo shows the workflow of the deployed application, including:

* Opening the web application
* Entering invoice information
* Predicting invoice risk
* Predicting freight cost
* Displaying machine learning predictions

For example:

```markdown
## Demo

🎥 ![Streamlit Application Demo]("C:\Users\91958\Downloads\Vendor Invoice Analysis\Project Video .mp4") 
```

---

# 🔮 Future Improvements

Potential improvements include:

* Batch CSV upload for invoice predictions
* Direct database-based prediction interface
* Real-time invoice monitoring
* Vendor-level risk scoring
* Fraud anomaly detection
* Automated retraining pipeline
* Feature engineering using vendor history
* Freight prediction using additional variables such as quantity, vendor information, and purchase history

---

# 👨‍💻 Author

**Satyam Chaurasiya**

This project demonstrates an end-to-end machine learning workflow combining:

```text
Data Engineering
      ↓
SQL Database
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Statistical Analysis
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Model Deployment
      ↓
Streamlit Web Application
```

---

## ⭐ If you found this project useful

Feel free to:

* ⭐ Star the repository
* Fork the project
* Open an issue
* Suggest improvements
* Connect for collaboration


# 🚗 AutoGuard AI

### Intelligent Vehicle Maintenance Risk Classification System

> **Predict vehicle maintenance risk using driving behavior, vehicle condition, usage patterns, and maintenance history with Machine Learning.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## 🌐 Project Overview

**AutoGuard AI** is a machine-learning-based vehicle maintenance risk classification system designed to analyze a vehicle's:

* 🚘 Vehicle characteristics
* 🛣️ Driving behavior
* 📊 Annual usage
* 🔧 Maintenance history
* 🌡️ Engine condition
* 🛞 Tire condition
* 🔋 Battery condition
* 🛑 Braking behavior

The system classifies a vehicle into three maintenance-risk categories:

| Risk Level    | Meaning                                                              |
| ------------- | -------------------------------------------------------------------- |
| 🟢 **Low**    | Lower maintenance risk based on the provided vehicle information     |
| 🟡 **Medium** | Moderate maintenance risk with some indicators requiring attention   |
| 🔴 **High**   | Higher maintenance risk with multiple indicators requiring attention |

The project uses **K-Nearest Neighbors (KNN)** as the main supervised machine-learning algorithm.

---

# 🎯 Problem Statement

Vehicle maintenance decisions are often based on mileage or vehicle age alone.

However, two vehicles with the same mileage can have very different maintenance conditions.

For example:

> A vehicle driven 20,000 km annually mostly on highways with regular servicing may experience a different maintenance risk from a vehicle driven the same distance in heavy city traffic with frequent hard braking and delayed servicing.

AutoGuard AI attempts to model this broader picture by combining:

**Vehicle Age + Usage + Driving Behavior + Vehicle Health + Maintenance History**

to classify the overall maintenance risk.

---

# 💡 Key Features

## 🤖 Machine Learning Classification

* K-Nearest Neighbors (KNN)
* Multi-class classification
* Automatic K-value tuning
* 5-fold cross-validation
* StandardScaler feature normalization
* Stratified train/test split
* Accuracy, Precision, Recall and F1 evaluation
* Confusion matrix

---

## 🚗 Vehicle Risk Prediction

Users can enter information such as:

* Vehicle age
* Vehicle type
* Annual mileage
* City driving percentage
* Daily driving hours
* Hard braking events
* Rapid acceleration events
* Engine temperature
* Engine vibration
* Oil age
* Brake wear
* Tire pressure
* Battery voltage
* Distance since last service
* Previous repairs
* Service frequency

The system then produces a maintenance-risk classification.

---

## 🧠 Smart Risk Analysis

AutoGuard AI does more than display the ML prediction.

It also identifies potential risk indicators such as:

* High vehicle age
* High annual mileage
* Excessive city driving
* Frequent hard braking
* Rapid acceleration
* Elevated engine temperature
* High engine vibration
* Old engine oil
* High brake wear
* Abnormal tire pressure
* Low battery voltage
* Long service intervals
* Frequent previous repairs
* Irregular maintenance

The system then provides maintenance-oriented recommendations based on these indicators.

> **Important:** The KNN model performs the classification, while the Smart Risk Analysis module provides rule-based explanations and recommendations.

---

# 📊 Interactive Dashboard

The Streamlit interface contains four main sections:

### 1. 📊 Dashboard

Provides an overview of:

* Total vehicles
* Low-risk vehicles
* Medium-risk vehicles
* High-risk vehicles
* Risk distribution
* Mileage vs engine vibration
* Dataset statistics

---

### 2. 🔍 Vehicle Prediction

Users can enter vehicle information and receive:

* Maintenance risk
* Model confidence distribution
* Detected risk factors
* Maintenance recommendations
* Vehicle analysis summary

---

### 3. 📈 Data Insights

Interactive visualizations explore relationships between vehicle characteristics and maintenance risk.

Examples include:

* Vehicle age vs risk
* Annual mileage vs risk
* Hard braking vs risk
* Rapid acceleration vs risk
* Engine vibration vs risk
* Engine temperature vs risk
* Oil age vs risk
* Brake wear vs risk
* City driving vs risk
* Previous repairs vs risk

---

### 4. 🤖 Model Performance

Displays:

* Test accuracy
* Precision
* Recall
* F1 score
* K-value tuning
* 5-fold cross-validation results
* Confusion matrix
* Model configuration

---

# 🧠 Machine Learning Pipeline

```text
                ┌──────────────────────┐
                │ Vehicle Dataset      │
                │ 15,000 Records       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Data Exploration     │
                │ EDA + Visualization  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Data Preprocessing   │
                │ Encoding + Cleaning  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Train/Test Split     │
                │ 80% / 20%            │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ StandardScaler       │
                │ Feature Scaling      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ KNN K-Value Tuning   │
                │ 5-Fold CV            │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ KNN Classifier       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Model Evaluation     │
                │ Accuracy / F1 / etc. │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Streamlit Interface  │
                └──────────────────────┘
```

---

# 📂 Project Structure

```text
Data classification using AI/
│
├── Data/
│   └── vehicle_data.csv
│
├── models/
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── model_info.pkl
│
├── notebooks/
│   └── 01_data_exploration.py
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── knn_model.py
│   ├── save_model.py
│   ├── prediction.py
│   ├── risk_analysis.py
│   └── test_prediction.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📋 Dataset

The project uses a **synthetic, domain-inspired dataset** containing:

### 15,000 vehicle records

The dataset includes vehicle usage, driving behavior, health indicators, and maintenance-history features.

### Main Features

| Feature                     | Description                         |
| --------------------------- | ----------------------------------- |
| `vehicle_age`               | Vehicle age in years                |
| `vehicle_type`              | Sedan, SUV, Hatchback or Pickup     |
| `annual_mileage`            | Approximate yearly mileage          |
| `city_driving_pct`          | Percentage of city driving          |
| `highway_driving_pct`       | Percentage of highway driving       |
| `daily_driving_hours`       | Average daily driving time          |
| `hard_braking_events`       | Estimated hard-braking events       |
| `rapid_acceleration_events` | Estimated rapid acceleration events |
| `engine_temperature`        | Engine operating temperature        |
| `engine_vibration`          | Engine vibration indicator          |
| `oil_age_km`                | Distance covered since oil change   |
| `brake_wear_pct`            | Estimated brake wear                |
| `tire_pressure_avg`         | Average tire pressure               |
| `battery_voltage`           | Battery voltage                     |
| `last_service_km`           | Distance since last service         |
| `previous_repairs`          | Number of previous repairs          |
| `service_frequency`         | Regular, Occasional or Rare         |
| `maintenance_risk`          | Target classification               |

---

# 🏷️ Target Variable

The target variable is:

```text
maintenance_risk
```

with three classes:

```text
Low
Medium
High
```

The classes are generated from a domain-inspired risk formulation using multiple vehicle condition and usage factors.

---

# 🧪 Data Generation

The dataset is generated programmatically using:

```python
NumPy
Pandas
```

This makes the project reproducible and allows a large dataset to be created for machine-learning experimentation.

The dataset is **synthetic** and should not be interpreted as a collection of real-world vehicle service records.

---

# ⚙️ Model Development

## Algorithm

The primary classification algorithm is:

### K-Nearest Neighbors (KNN)

KNN classifies a new vehicle based on the characteristics of nearby training examples.

Before KNN training, numerical features are standardized using:

```python
StandardScaler
```

Categorical variables are converted using:

```python
pd.get_dummies()
```

---

# 🔎 K-Value Optimization

Instead of manually selecting K, the project evaluates multiple odd K values:

```text
1, 3, 5, 7, ..., 21
```

Each value is evaluated using:

### 5-Fold Cross-Validation

The K value with the highest mean validation accuracy is selected for the final KNN model.

This helps avoid selecting K solely from the held-out test set.

---

# 📏 Model Evaluation

The project evaluates the classifier using:

### Accuracy

Overall percentage of correct predictions.

### Precision

Measures how often predicted classes are correct.

### Recall

Measures how effectively the model identifies the actual classes.

### F1 Score

Provides a combined measure of precision and recall.

### Confusion Matrix

Shows how predictions are distributed across:

```text
Low
Medium
High
```

---

# 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Pandas       | Data manipulation         |
| NumPy        | Numerical computation     |
| Scikit-learn | Machine learning          |
| Matplotlib   | Data visualization        |
| Seaborn      | Statistical visualization |
| Plotly       | Interactive charts        |
| Streamlit    | Web application           |
| Joblib       | Model serialization       |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

```bash
cd "Data classification using AI"
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🔬 Run the ML Components

### Generate Dataset

```bash
python src/data_generator.py
```

### Explore Dataset

```bash
python notebooks/01_data_exploration.py
```

### Train and Evaluate KNN

```bash
python src/knn_model.py
```

### Save Trained Model

```bash
python src/save_model.py
```

### Test Prediction

```bash
python src/test_prediction.py
```

---

# 💾 Saved Model Files

The trained model components are stored in:

```text
models/
```

### `knn_model.pkl`

Stores the trained KNN classifier.

### `scaler.pkl`

Stores the fitted StandardScaler.

### `feature_columns.pkl`

Stores the feature order used during model training.

### `model_info.pkl`

Stores model configuration and training information.

---

# 🖥️ Application Workflow

```text
User enters vehicle information
            │
            ▼
Feature preparation
            │
            ▼
Categorical encoding
            │
            ▼
Feature alignment
            │
            ▼
StandardScaler
            │
            ▼
Trained KNN model
            │
            ▼
Maintenance Risk
            │
       ┌────┼────┐
       ▼    ▼    ▼
      Low Medium High
            │
            ▼
    Smart Risk Analysis
            │
            ▼
Maintenance Recommendations
```

---

# 📸 Screenshots

Add screenshots of the application here after deployment.

### Dashboard

```text
![Dashboard](screenshots/dashboard.png)
```

### Vehicle Prediction

```text
![Vehicle Prediction](screenshots/prediction.png)
```

### Data Insights

```text
![Data Insights](screenshots/insights.png)
```

### Model Performance

```text
![Model Performance](screenshots/model-performance.png)
```

> Create a `screenshots/` folder in the repository and place your application screenshots inside it.

---

# 🌐 Live Demo

After deployment, add your Streamlit application link here:

**🔗 Live Demo:** `YOUR_STREAMLIT_APP_URL`

---

# 🔗 Project Repository

**GitHub:** `YOUR_GITHUB_REPOSITORY_URL`

---

# 🧠 What This Project Demonstrates

This project demonstrates practical understanding of:

* Supervised Machine Learning
* Multi-class classification
* K-Nearest Neighbors
* Feature engineering
* Categorical encoding
* Feature scaling
* Train/test splitting
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* Data visualization
* Interactive dashboards
* Model serialization
* Python project structure
* Streamlit application development
* Git/GitHub workflow

---

# 🔮 Future Improvements

Possible future improvements include:

* Real-world vehicle maintenance datasets
* Larger real-world datasets
* Additional classification algorithms
* Model comparison
* Explainable AI techniques
* Time-series maintenance prediction
* Vehicle sensor/IoT integration
* Automated maintenance reminders
* User accounts and vehicle history
* PDF maintenance reports
* Cloud database integration
* Mobile-friendly interface
* Predictive component-level failure analysis

---

# ⚠️ Limitations

AutoGuard AI is an educational machine-learning project.

The current dataset is **synthetic and domain-inspired**, not collected from actual vehicle service centers or vehicle sensors.

Therefore:

* Predictions should not be treated as professional mechanical diagnosis.
* Model performance on this dataset does not represent real-world vehicle-failure prediction accuracy.
* Maintenance recommendations are informational.
* Real-world deployment would require validated automotive datasets and domain expertise.

---

# 🎓 Project Type

**Machine Learning / Artificial Intelligence Project**

### Core Concept

```text
Classification of Data using AI
```

### Learning Type

```text
Supervised Learning
```

### Algorithm

```text
K-Nearest Neighbors (KNN)
```

### Classification

```text
Multi-Class Classification
```

### Target

```text
Vehicle Maintenance Risk
```

---

# 👨‍💻 Author

## Sarmad Amin

BS Computer Science Student
University of Central Punjab — Lahore, Pakistan

### Skills Demonstrated in This Project

`Python` • `Pandas` • `NumPy` • `Scikit-learn` • `KNN` • `Data Visualization` • `Streamlit` • `Machine Learning` • `GitHub`

---

# ⭐ If You Find This Project Interesting

Feel free to explore the code, experiment with the model, and improve the system with real-world datasets and additional machine-learning techniques.

---

## 📜 License

This project is available under the **MIT License**.

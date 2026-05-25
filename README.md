# MedicoBuddy — Disease Prediction Web Application

A multi-disease prediction web application built with Flask and Scikit-learn that provides probability-based assessments for COVID-19, Diabetes, and Heart Disease using trained machine learning models. Features an interactive web interface with form-based symptom input and real-time prediction results, designed for deployment on Heroku.

## 📌 Overview

MedicoBuddy enables users to input their health parameters through a clean, responsive web interface and receive instant disease risk predictions powered by pre-trained ML models. The application currently supports three prediction modules with plans for expansion.

## 🩺 Prediction Modules

### 1. COVID-19 Probability Detector
- **Model:** Logistic Regression
- **Input Features:** Fever, Age, Body Pain, Runny Nose, Difficulty Breathing, Chest Pain
- **Output:** Infection probability percentage

### 2. Diabetes Predictor
- **Model:** Random Forest Classifier
- **Input Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Output:** Diabetes prediction (Positive/Negative)

### 3. Heart Disease Predictor
- **Model:** Logistic Regression
- **Input Features:** Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Induced Angina, Oldpeak, Slope, CA, Thal
- **Output:** Heart disease prediction (Positive/Negative)

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML5, Bootstrap 5
- **ML Libraries:** Scikit-learn, Pandas, NumPy
- **Model Serialization:** Pickle
- **Deployment:** Heroku (Gunicorn)
- **Chatbot:** Dialogflow Integration

## 📊 Datasets

| Module | Dataset | Description |
|--------|---------|-------------|
| COVID-19 | `data.csv` | Symptom-based infection probability data |
| Diabetes | `diabetes.csv` | Pima Indians Diabetes dataset |
| Heart Disease | `heart.csv` | Heart disease clinical records |

## 📁 Project Structure

```
covid-19-prediction/
├── main.py                              # Flask application with all routes
├── training.py                          # COVID-19 model training script
├── solution.ipynb                       # COVID-19 data analysis notebook
├── diabetes.ipynb                       # Diabetes model training notebook
├── heart.ipynb                          # Heart disease model training notebook
├── model.pkl                            # COVID-19 trained model
├── diabetes-prediction-rfc-model.pkl    # Diabetes trained model
├── heart.pkl                            # Heart disease trained model
├── insurance.pkl                        # Insurance cost predictor model
├── data.csv                             # COVID-19 dataset
├── diabetes.csv                         # Diabetes dataset
├── heart.csv                            # Heart disease dataset
├── requirements.txt                     # Python dependencies
├── Procfile                             # Heroku deployment config
├── runtime.txt                          # Python runtime version
├── templates/                           # HTML templates
│   ├── index.html                       # Home page
│   ├── covid.html                       # COVID-19 input form
│   ├── show.html                        # COVID-19 results
│   ├── diabetes.html                    # Diabetes input form
│   ├── diashow.html                     # Diabetes results
│   ├── heart.html                       # Heart disease input form
│   └── heartshow.html                   # Heart disease results
└── static/
    └── styles/
        └── style.css                    # Custom styles
```

## 🚀 Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/jigyasavats/covid-19-prediction.git
   cd covid-19-prediction
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application
   ```bash
   python main.py
   ```
4. Open `http://127.0.0.1:5000` in your browser

## 📝 Key Highlights

- Multi-disease prediction platform with modular architecture
- Pre-trained ML models serialized using Pickle for fast inference
- Responsive Bootstrap 5 UI with form validation
- Heroku-ready deployment with Gunicorn and Procfile
- Dialogflow chatbot integration for interactive user assistance

# Heart Disease Diagnosis Using Machine Learning and Data Mining

This repository contains the code, notebook, trained model, dataset, and dissertation report for a third-year Data Analytics project on heart disease prediction. The project applies supervised machine learning techniques to clinical patient data and compares multiple classifiers to identify the most effective model for diagnosis support.

## Project Overview

Heart disease remains one of the leading causes of death worldwide, so early prediction can support faster and better clinical decisions. This project builds a machine learning pipeline that:

- explores and visualizes a heart disease dataset,
- preprocesses patient records for modeling,
- compares multiple classification algorithms,
- evaluates performance with standard metrics, and
- provides a simple Tkinter GUI for individual patient prediction.

## Objectives

- Analyse clinical features associated with heart disease.
- Compare the performance of Decision Tree, Random Forest, Naive Bayes, and Support Vector Machine models.
- Identify influential predictors using aggregated feature importance.
- Save the best-performing model for reuse.
- Provide a user-friendly prediction interface for manual input.

## Dataset

The dataset used in this project is `dataset/heart_statlog_cleveland_hungary_final.csv`.

### Dataset Summary

- Records: 1,190
- Features: 11 clinical input variables
- Target classes: 2
- Heart disease cases (`1`): 629
- No heart disease cases (`0`): 561
- Missing values in the current CSV: 0

### Features Used

The model uses the following clinical attributes:

- `age`
- `sex`
- `chest pain type`
- `resting bp s`
- `cholesterol`
- `fasting blood sugar`
- `resting ecg`
- `max heart rate`
- `exercise angina`
- `oldpeak`
- `ST slope`

The target variable is `target`, where:

- `0` = no heart disease
- `1` = heart disease

## Methodology

### 1. Data Preprocessing

- Load the dataset with Pandas.
- Check for missing values.
- Replace the `ST slope` outlier value `0` with the column mode.
- Split the data into training and test sets using a 75:25 ratio.
- Standardize the feature values using `StandardScaler`.

### 2. Exploratory Data Analysis

The project includes visual analysis of:

- class distribution,
- categorical feature relationships with the target,
- feature importance ranking,
- model accuracy comparison,
- confusion matrices,
- ROC curves,
- precision-recall curves, and
- PCA-based decision boundary plots.

### 3. Models Evaluated

- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Support Vector Machine

### 4. Feature Ranking

Feature importance is aggregated across the evaluated models. In the current implementation, the strongest predictors ranked by importance include:

- `ST slope`
- `chest pain type`
- `oldpeak`
- `cholesterol`
- `sex`

## Results

The following results were reproduced from the current codebase using the same preprocessing logic, `random_state=0`, a 25% test split, and the selected ranked features:

| Model                  | Test Accuracy | ROC-AUC | Mean 5-Fold CV Accuracy |
| ---------------------- | ------------: | ------: | ----------------------: |
| Decision Tree          |        0.8725 |  0.8732 |                  0.8576 |
| Random Forest          |        0.9497 |  0.9628 |                  0.8980 |
| Naive Bayes            |        0.8591 |  0.9006 |                  0.8318 |
| Support Vector Machine |        0.8322 |  0.8823 |                  0.8408 |

### Best Performing Model

`Random Forest` achieved the strongest overall performance in this project and is the model saved to `random_forest_model.pkl` for GUI-based prediction.

## GUI Prediction System

The Python script includes a Tkinter-based desktop interface where a user can enter patient values and receive a prediction result.

### Input Encoding Used in the GUI

- `Sex`: `1 = Male`, `0 = Female`
- `Chest Pain Type`: values `1` to `4`
- `Fasting Blood Sugar`: `1 = true`, `0 = false`
- `Resting ECG`: values `0` to `2`
- `Exercise Induced Angina`: `1 = Yes`, `0 = No`
- `ST Slope`: values `1` to `3`

## Project Structure

```text
Heart-Disease-Diagnosis-using-Machine-Learning-and-Data-Mining/
├── dataset/
│   └── heart_statlog_cleveland_hungary_final.csv
├── doc/
│   └── 10892938 - PROJ 518 FINAL DISSERTATION.pdf
├── script/
│   ├── 10892938 - Project_Code.ipynb
│   └── 10892938 - Project_Code.py
├── random_forest_model.pkl
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Tkinter
- Jupyter Notebook

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Flask Prediction Model

```bash
python train_model.py
```

This generates a trained pipeline at `model/heart_disease_pipeline.pkl`.

### 3. Start the Flask Web App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser to use the web UI.

### 4. Deploy to Render

Render can build and serve the app automatically. Add the service on Render, then set the build command to:

```bash
python train_model.py
```

and the start command to:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

Render will use the generated `model/heart_disease_pipeline.pkl` when serving the app.

### 4. Optional: Run the original analysis script

```bash
python "script/10892938 - Project_Code.py"
```

This will still load and preprocess the dataset, train the Random Forest model, and launch the original Tkinter interface.

### 5. Project Structure for Flask Deployment

```text
Heart-Disease-Diagnosis-Using-Machine-Learning-and-Data-Mining/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
│   └── heart_statlog_cleveland_hungary_final.csv
├── model/
│   └── heart_disease_pipeline.pkl  # created after train_model.py runs
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── random_forest_model.pkl
├── script/
│   ├── 10892938 - Project_Code.ipynb
│   └── 10892938 - Project_Code.py
└── README.md
```

### Notes

- The Flask app uses the same dataset and a `RandomForestClassifier` pipeline with `StandardScaler`.
- The web UI is designed for screening only and should not replace clinical judgment.
- For production, configure a secure `SECRET_KEY` and use a production WSGI server such as Gunicorn or uWSGI.

## Academic Value

This project demonstrates core Data Analytics and machine learning skills, including:

- data cleaning and preprocessing,
- exploratory data analysis,
- feature engineering and ranking,
- classification model comparison,
- performance evaluation, and
- deployment of a simple interactive application.

## Future Improvements

- Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
- Package preprocessing and modeling into a single reusable pipeline.
- Add model explainability with SHAP or LIME.
- Build a web-based interface using Flask or Streamlit.
- Add automated tests and a `requirements.txt` file for easier setup.

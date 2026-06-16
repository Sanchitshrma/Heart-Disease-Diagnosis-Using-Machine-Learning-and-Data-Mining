import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'dataset' / 'heart_statlog_cleveland_hungary_final.csv'
MODEL_DIR = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'heart_disease_pipeline.pkl'

FEATURE_COLUMNS = [
    'age',
    'sex',
    'chest pain type',
    'resting bp s',
    'cholesterol',
    'fasting blood sugar',
    'resting ecg',
    'max heart rate',
    'exercise angina',
    'oldpeak',
    'ST slope'
]


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df['ST slope'] = df['ST slope'].replace(0, df['ST slope'].mode()[0])
    return df


def build_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])


def train():
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df['target']

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f'Saved trained pipeline to {MODEL_PATH}')


if __name__ == '__main__':
    train()

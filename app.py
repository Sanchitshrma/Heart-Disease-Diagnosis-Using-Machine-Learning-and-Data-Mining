import os
import sys
import joblib
from pathlib import Path
from flask import Flask, render_template, request, flash

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model' / 'heart_disease_pipeline.pkl'

FEATURE_CONFIGS = [
    {'key': 'age', 'label': 'Age', 'type': 'number', 'step': '1', 'placeholder': '40'},
    {'key': 'sex', 'label': 'Sex', 'type': 'select', 'choices': [('Male', 1), ('Female', 0)]},
    {'key': 'cp', 'label': 'Chest Pain Type', 'type': 'select', 'choices': [
        ('1 - Typical angina', 1),
        ('2 - Atypical angina', 2),
        ('3 - Non-anginal pain', 3),
        ('4 - Asymptomatic', 4)
    ]},
    {'key': 'trestbps', 'label': 'Resting Blood Pressure (mm Hg)', 'type': 'number', 'step': '1', 'placeholder': '140'},
    {'key': 'chol', 'label': 'Cholesterol (mg/dL)', 'type': 'number', 'step': '1', 'placeholder': '289'},
    {'key': 'fbs', 'label': 'Fasting Blood Sugar', 'type': 'select', 'choices': [
        ('120 mg/dL or below', 0),
        ('Above 120 mg/dL', 1)
    ]},
    {'key': 'restecg', 'label': 'Resting ECG', 'type': 'select', 'choices': [
        ('0 - Normal', 0),
        ('1 - ST-T abnormality', 1),
        ('2 - LVH', 2)
    ]},
    {'key': 'thalach', 'label': 'Max Heart Rate', 'type': 'number', 'step': '1', 'placeholder': '172'},
    {'key': 'exang', 'label': 'Exercise Induced Angina', 'type': 'select', 'choices': [
        ('No', 0),
        ('Yes', 1)
    ]},
    {'key': 'oldpeak', 'label': 'Oldpeak', 'type': 'number', 'step': '0.1', 'placeholder': '0.0'},
    {'key': 'slope', 'label': 'ST Slope', 'type': 'select', 'choices': [
        ('1 - Upsloping', 1),
        ('2 - Flat', 2),
        ('3 - Downsloping', 3)
    ]}
]

FIELD_TO_COLUMN = {
    'age': 'age',
    'sex': 'sex',
    'cp': 'chest pain type',
    'trestbps': 'resting bp s',
    'chol': 'cholesterol',
    'fbs': 'fasting blood sugar',
    'restecg': 'resting ecg',
    'thalach': 'max heart rate',
    'exang': 'exercise angina',
    'oldpeak': 'oldpeak',
    'slope': 'ST slope'
}

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'change-this-secret-key'


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            'Trained model not found. Run `python train_model.py` first to generate the model.'
        )
    return joblib.load(MODEL_PATH)

model = load_model()


def format_prediction(prediction, probability):
    if prediction == 1:
        return {
            'label': 'Heart disease likely',
            'message': 'The model indicates a higher probability of heart disease. This is a screening estimate and should be confirmed by a clinician.',
            'risk': f'{probability:.1%}',
            'confidence': f'{probability:.1%}',
            'status': 'high'
        }
    return {
        'label': 'Heart disease unlikely',
        'message': 'The model indicates a lower probability of heart disease. This is a screening estimate and not a diagnosis.',
        'risk': f'{probability:.1%}',
        'confidence': f'{probability:.1%}',
        'status': 'low'
    }


def parse_input(form):
    values = []
    for field in FEATURE_CONFIGS:
        key = field['key']
        raw = form.get(key, '').strip()
        if raw == '':
            raise ValueError(f'Missing value for {field["label"]}.')

        if field['type'] == 'select':
            values.append(int(raw))
            continue

        if field['type'] == 'number':
            value = float(raw) if field['step'] and '.' in field['step'] else int(raw)
            values.append(value)
            continue

    return values


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    form_data = {field['key']: '' for field in FEATURE_CONFIGS}

    if request.method == 'POST':
        try:
            for field in FEATURE_CONFIGS:
                form_data[field['key']] = request.form.get(field['key'], '').strip()

            values = parse_input(request.form)
            prediction = model.predict([values])[0]
            probability = float(model.predict_proba([values])[0][1])
            prediction_result = format_prediction(prediction, probability)
        except ValueError as error:
            flash(str(error), 'error')
        except Exception as error:
            flash('Unable to generate a prediction. Please verify all inputs.', 'error')
            app.logger.exception(error)

    return render_template(
        'index.html',
        fields=FEATURE_CONFIGS,
        form_data=form_data,
        prediction=prediction_result
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json or {}
    try:
        values = [int(data[k]) if field['type'] == 'select' else float(data[k]) if '.' in field['step'] else int(data[k])
                  for field in FEATURE_CONFIGS for k in [field['key']]]
        prediction = model.predict([values])[0]
        probability = float(model.predict_proba([values])[0][1])
        result = format_prediction(prediction, probability)
        return {
            'prediction': int(prediction),
            'probability': probability,
            'result': result
        }, 200
    except Exception as error:
        return {'error': 'Invalid input payload or prediction error.'}, 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print('Port must be a number.', file=sys.stderr)
            sys.exit(1)

    debug_flag = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    app.run(host='0.0.0.0', port=port, debug=debug_flag)

#!/usr/bin/env python
# coding: utf-8

# # PROJECT DISSERTATION CODE

# ### Importing the Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


# ### Importing the dataset

# In[ ]:


# Resolve paths relative to the project so the script works from any cwd.
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / 'dataset' / 'heart_statlog_cleveland_hungary_final.csv'
MODEL_PATH = BASE_DIR / 'random_forest_model.pkl'

dataset = pd.read_csv(DATASET_PATH)
print('The shape of the data is ', dataset.shape)


# ### A preview of the dataset

# In[ ]:


print(dataset.head(20))


# ### Dividing the features into Numerical and Categorical

# In[ ]:


col = list(dataset.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(dataset[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)


# ### Mapping the categorical features to their corresponding labels

# In[ ]:


df1 = dataset.copy()
df1['sex'] = df1['sex'].replace({1: 'male', 0: 'female'})
df1['chest pain type'] = df1['chest pain type'].replace({1: 'typical ang', 2: 'atypical ang', 3: 'non-ang', 4: 'asymptomatic'})
df1['fasting blood sugar'] = df1['fasting blood sugar'].replace({1: 'true', 0: 'false'})
df1['resting ecg'] = df1['resting ecg'].replace({0: 'normal', 1: 'ST-T abnormality', 2: 'LVH'})
df1['exercise angina'] = df1['exercise angina'].replace({1: 'yes', 0: 'no'})
df1['ST slope'] = df1['ST slope'].replace({1: 'upsloping', 2: 'flat', 3: 'downsloping'})
df1['target'] = df1['target'].replace({1: 'heart disease', 0: 'no heart disease'})


# ### Data Preprocessing

# In[ ]:


# Data Preprocessing function
def data_preprocessing(dataset):
    # Handle missing values
    missing_values = dataset.isnull().sum()
    
    # Replace the outlier with the mode of the column
    mode_st_slope = dataset['ST slope'].mode()[0]
    dataset['ST slope'] = dataset['ST slope'].replace(0, mode_st_slope)
    print("Outlier replaced with mode:", mode_st_slope)
    
    # Split the data into features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, missing_values, scaler

# Perform data preprocessing
X_train, X_test, y_train, y_test, missing_values, scaler = data_preprocessing(dataset)

# Print dataset information and missing values
print("The dataset information:")
print(dataset.info())
print("\nMissing values in each column:")
print(missing_values)


# ### Feature Selection, Training and Evaluation of the Models

# In[ ]:


def run_full_analysis():
    target_colors = ['#2ecc71', '#f7342a']

    d = list(df1['target'].value_counts())
    circle = [d[1] / sum(d) * 100, d[0] / sum(d) * 100]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.pie(
        circle,
        labels=['Normal', 'Heart Disease'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.1, 0),
        colors=target_colors,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True},
        textprops={'fontsize': 14}
    )
    plt.title('Heart Disease (%)', fontsize=18)

    plt.subplot(1, 2, 2)
    ax = sns.countplot(x='target', data=df1, palette=target_colors, edgecolor='black')
    for rect in ax.patches:
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 2,
            rect.get_height(),
            horizontalalignment='center',
            fontsize=14
        )
    ax.set_xticklabels(['Normal', 'Heart Disease'], fontsize=14)
    ax.set_xlabel('Target', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    plt.title('Cases of Heart Disease', fontsize=18)
    plt.show()

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for i in range(len(categorical_features) - 1):
        plt.subplot(3, 2, i + 1)
        ax = sns.countplot(
            x=categorical_features[i],
            data=df1,
            hue="target",
            palette=target_colors,
            edgecolor='black'
        )
        for rect in ax.patches:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 2,
                rect.get_height(),
                horizontalalignment='center',
                fontsize=11
            )
        title = categorical_features[i] + ' vs Heart Disease'
        plt.legend(['Normal', 'Heart Disease'])
        plt.title(title)

    plt.tight_layout()
    plt.show()

    # Initialize models to evaluate
    models = {
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
        'Random Forest': RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(kernel='linear', random_state=0, probability=True)
    }

    # Dictionary to store feature importances for each model
    feature_importance_info = {model_name: None for model_name in models.keys()}

    # Evaluate models and get feature importances
    for model_name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):  # For models like SVM with coefficients
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            importances = None

        feature_importance_info[model_name] = importances

    # Aggregate feature importances across models
    feature_names = dataset.columns[:-1]
    aggregate_importances = np.zeros(len(feature_names))
    for model_name, importances in feature_importance_info.items():
        if importances is not None:
            aggregate_importances += importances

    # Normalize aggregated importances
    aggregate_importances /= len(models)

    # Sort indices based on aggregated importances
    indices = np.argsort(aggregate_importances)[::-1]

    # Create DataFrame to display aggregated feature importances
    aggregate_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': aggregate_importances})
    aggregate_importance_df = aggregate_importance_df.sort_values(by='Importance', ascending=False)

    print("\nAggregate Feature Importances Across Models:")
    print(aggregate_importance_df)

    def evaluate_model_with_selected_features(model, X_train, X_test, y_train, y_test, indices, num_features):
        top_features = indices[:num_features]
        X_train_selected = X_train[:, top_features]
        X_test_selected = X_test[:, top_features]

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        return accuracy_score(y_test, y_pred)

    results = {model_name: [] for model_name in models.keys()}
    for num_features in range(1, len(indices) + 1):
        for model_name, model in models.items():
            accuracy = evaluate_model_with_selected_features(
                model, X_train, X_test, y_train, y_test, indices, num_features
            )
            results[model_name].append(accuracy)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=aggregate_importance_df, palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    plt.figure(figsize=(12, 8))
    for model_name, accuracies in results.items():
        plt.plot(range(1, len(indices) + 1), accuracies, label=model_name, marker='o')
    plt.title('Model Performance with Increasing Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    best_accuracy_info = {model_name: {'accuracy': 0, 'num_features': 0} for model_name in models.keys()}
    for num_features in range(1, len(indices) + 1):
        for model_name, model in models.items():
            accuracy = evaluate_model_with_selected_features(
                model, X_train, X_test, y_train, y_test, indices, num_features
            )
            if accuracy > best_accuracy_info[model_name]['accuracy']:
                best_accuracy_info[model_name]['accuracy'] = accuracy
                best_accuracy_info[model_name]['num_features'] = num_features

    for model_name, info in best_accuracy_info.items():
        print(f'Model: {model_name}, Highest Accuracy: {info["accuracy"]}, Number of Features: {info["num_features"]}')

    top_n = 11
    top_features = indices[:top_n]
    X_train_selected = X_train[:, top_features]
    X_test_selected = X_test[:, top_features]

    model_names = []
    accuracies = []
    conf_matrices = []
    class_reports = []
    precision_recall_curves = []
    roc_auc_scores = []

    for name, model in models.items():
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        print(f'{name} Model with Top {top_n} Features:')

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print('Accuracy:', accuracy)

        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_reports.append(class_report)
        print('Classification Report:\n', classification_report(y_test, y_pred))

        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrices.append(conf_matrix)
        print('Confusion Matrix:\n', conf_matrix)

        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        precision_recall_curves.append((precision, recall))

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test_selected)[:, 1]
        else:
            y_scores = model.decision_function(X_test_selected)

        roc_auc = roc_auc_score(y_test, y_scores)
        roc_auc_scores.append(roc_auc)
        print('ROC_AUC Score:\n', roc_auc_score(y_test, y_scores))

        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
        print(f'{name} Cross-Validation Scores:', cv_scores)
        print(f'{name} Mean CV Accuracy: {np.mean(cv_scores)}')
        print('\n' + '-' * 30 + '\n')

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

        model_names.append(name)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    for name, model in models.items():
        if hasattr(model, 'support_'):
            num_support_vectors_per_class = model.n_support_
            print(f'Number of support vectors for each class: {num_support_vectors_per_class}')
        else:
            print(f'{name} does not use support vectors.')

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    def plot_decision_boundary(model, X, y, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
        plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        legend1 = plt.legend(*scatter.legend_elements(), title="Targets")
        plt.gca().add_artist(legend1)
        plt.show()

    for name, model in models.items():
        model.fit(X_train_pca, y_train)
        plot_decision_boundary(model, X_train_pca, y_train, f'{name} Decision Boundary (Train)')
        plot_decision_boundary(model, X_test_pca, y_test, f'{name} Decision Boundary (Test)')

    plt.figure(figsize=(10, 6))
    for i, (precision, recall) in enumerate(precision_recall_curves):
        plt.plot(recall, precision, label=model_names[i])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    accuracy_colors = sns.color_palette("colorblind", len(model_names))
    plt.figure(figsize=(12, 7))
    bars = plt.bar(model_names, accuracies, width=0.6, color=accuracy_colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f'{yval:.2f}',
            ha='center',
            va='bottom',
            fontsize=12,
            color='black'
        )

    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Accuracy Comparison of the Models', fontsize=18, fontweight='bold', color='#4f4f4f')
    plt.ylim([0, 1.1])
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    confusion_colors = ['#7971e3', '#eb8e46']
    font_size = 14

    for i, (name, conf_matrix) in enumerate(zip(model_names, conf_matrices)):
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            ax=axes[i],
            cmap=confusion_colors,
            cbar=False,
            annot_kws={"size": font_size}
        )
        axes[i].set_title(f'{name} Confusion Matrix', fontsize=font_size)
        axes[i].set_xlabel('Predicted', fontsize=font_size)
        axes[i].set_ylabel('Actual', fontsize=font_size)
        axes[i].tick_params(axis='both', which='major', labelsize=font_size)

    plt.tight_layout()
    plt.show()

    precision_class_0 = [report['0']['precision'] for report in class_reports]
    precision_class_1 = [report['1']['precision'] for report in class_reports]
    recall_class_0 = [report['0']['recall'] for report in class_reports]
    recall_class_1 = [report['1']['recall'] for report in class_reports]
    f1_class_0 = [report['0']['f1-score'] for report in class_reports]
    f1_class_1 = [report['1']['f1-score'] for report in class_reports]
    comparison_model_names = ["Decision Tree", "Random Forest", "Naive Bayes", "Support Vector Machine"]

    def plot_precision_by_class(model_names, precision_class_0, precision_class_1):
        bar_width = 0.35
        index = np.arange(len(model_names))

        plt.figure(figsize=(12, 7))
        bars1 = plt.bar(index, precision_class_0, bar_width, label='Class 0')
        bars2 = plt.bar(index + bar_width, precision_class_1, bar_width, label='Class 1')

        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Precision by Class for Each Model', fontsize=18, fontweight='bold', color='#4f4f4f')
        plt.xticks(index + bar_width / 2, model_names)
        plt.ylim([0, 1.1])
        plt.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.01,
                    f'{yval:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    color='black'
                )

        plt.gca().yaxis.grid(True, linestyle='--', alpha=0.6)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_recall_by_class(model_names, recall_class_0, recall_class_1):
        bar_width = 0.35
        index = np.arange(len(model_names))

        plt.figure(figsize=(12, 7))
        bars1 = plt.bar(index, recall_class_0, bar_width, label='Class 0')
        bars2 = plt.bar(index + bar_width, recall_class_1, bar_width, label='Class 1')

        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.ylabel('Recall', fontsize=14, fontweight='bold')
        plt.title('Recall by Class for Each Model', fontsize=18, fontweight='bold', color='#4f4f4f')
        plt.xticks(index + bar_width / 2, model_names)
        plt.ylim([0, 1.1])
        plt.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.01,
                    f'{yval:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    color='black'
                )

        plt.gca().yaxis.grid(True, linestyle='--', alpha=0.6)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_f1_by_class(model_names, f1_class_0, f1_class_1):
        bar_width = 0.35
        index = np.arange(len(model_names))

        plt.figure(figsize=(12, 7))
        bars1 = plt.bar(index, f1_class_0, bar_width, label='Class 0')
        bars2 = plt.bar(index + bar_width, f1_class_1, bar_width, label='Class 1')

        plt.xlabel('Model', fontsize=14, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
        plt.title('F1 Score by Class for Each Model', fontsize=18, fontweight='bold', color='#4f4f4f')
        plt.xticks(index + bar_width / 2, model_names)
        plt.ylim([0, 1.1])
        plt.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.01,
                    f'{yval:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    color='black'
                )

        plt.gca().yaxis.grid(True, linestyle='--', alpha=0.6)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    plot_precision_by_class(comparison_model_names, precision_class_0, precision_class_1)
    plot_recall_by_class(comparison_model_names, recall_class_0, recall_class_1)
    plot_f1_by_class(comparison_model_names, f1_class_0, f1_class_1)


# ### Graphical User Interface

# In[ ]:


rf_model = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0)
rf_model.fit(X_train, y_train)

# Save the trained Random Forest model
with open(MODEL_PATH, 'wb') as model_file:
    joblib.dump(rf_model, model_file)
    
# Load the trained Random Forest model
with open(MODEL_PATH, 'rb') as model_file:
    random_model = joblib.load(model_file)

APP_BG = '#07111d'
PANEL_BG = '#0f1c2c'
SURFACE_BG = '#12263a'
FIELD_BG = '#091522'
FIELD_BORDER = '#29435f'
TEXT_PRIMARY = '#f4efe6'
TEXT_SECONDARY = '#bed0dd'
TEXT_MUTED = '#7890a2'
ACCENT = '#ff8a5c'
ACCENT_SOFT = '#ffd8c9'
SUCCESS = '#35c58b'
SUCCESS_SOFT = '#c8f3e2'
WARNING = '#ff6b6b'
WARNING_SOFT = '#ffd8d8'
GOLD = '#f3c96b'
BUTTON_TEXT_DARK = '#08131f'
EMPTY_SELECT = 'Choose an option'

FIELD_DEFINITIONS = [
    {'key': 'age', 'label': 'Age', 'kind': 'entry', 'parser': int, 'hint': 'Years'},
    {'key': 'sex', 'label': 'Sex', 'kind': 'combo', 'hint': 'Dataset encoding handled for you',
     'choices': [('Male', 1), ('Female', 0)]},
    {'key': 'cp', 'label': 'Chest Pain Type', 'kind': 'combo', 'hint': 'Select the pain presentation',
     'choices': [('1 - Typical angina', 1), ('2 - Atypical angina', 2), ('3 - Non-anginal pain', 3), ('4 - Asymptomatic', 4)]},
    {'key': 'trestbps', 'label': 'Resting Blood Pressure', 'kind': 'entry', 'parser': int, 'hint': 'mm Hg'},
    {'key': 'chol', 'label': 'Cholesterol', 'kind': 'entry', 'parser': int, 'hint': 'mg/dL'},
    {'key': 'fbs', 'label': 'Fasting Blood Sugar', 'kind': 'combo', 'hint': 'Blood sugar threshold at 120 mg/dL',
     'choices': [('Above 120 mg/dL', 1), ('120 mg/dL or below', 0)]},
    {'key': 'restecg', 'label': 'Resting ECG', 'kind': 'combo', 'hint': 'Resting electrocardiographic result',
     'choices': [('0 - Normal', 0), ('1 - ST-T abnormality', 1), ('2 - LVH', 2)]},
    {'key': 'thalach', 'label': 'Max Heart Rate', 'kind': 'entry', 'parser': int, 'hint': 'Beats per minute'},
    {'key': 'exang', 'label': 'Exercise Induced Angina', 'kind': 'combo', 'hint': 'Symptoms observed during exercise',
     'choices': [('Yes', 1), ('No', 0)]},
    {'key': 'oldpeak', 'label': 'Oldpeak', 'kind': 'entry', 'parser': float, 'hint': 'ST depression induced by exercise'},
    {'key': 'slope', 'label': 'ST Slope', 'kind': 'combo', 'hint': 'Slope of the peak exercise ST segment',
     'choices': [('1 - Upsloping', 1), ('2 - Flat', 2), ('3 - Downsloping', 3)]}
]

SAMPLE_VALUES = {
    'age': '40',
    'sex': 'Male',
    'cp': '2 - Atypical angina',
    'trestbps': '140',
    'chol': '289',
    'fbs': '120 mg/dL or below',
    'restecg': '0 - Normal',
    'thalach': '172',
    'exang': 'No',
    'oldpeak': '0.0',
    'slope': '1 - Upsloping'
}

choice_maps = {}
input_vars = {}
input_widgets = {}


def set_result_panel_state(title, message, probability_text, confidence_text, accent_color, accent_soft, badge_text):
    result_badge_label.configure(text=badge_text, bg=accent_soft, fg=accent_color)
    result_card.configure(highlightbackground=accent_color)
    result_title_var.set(title)
    result_message_var.set(message)
    probability_var.set(probability_text)
    confidence_var.set(confidence_text)


def reset_result_panel():
    risk_meter.configure(style='Neutral.Horizontal.TProgressbar')
    risk_meter['value'] = 0
    set_result_panel_state(
        'Prediction will appear here',
        'Complete the patient profile and press Predict to evaluate the model.',
        'Estimated heart disease probability: --',
        'Model confidence: --',
        GOLD,
        '#2c3e50',
        'Ready'
    )


def clear_form():
    for config in FIELD_DEFINITIONS:
        default_value = EMPTY_SELECT if config['kind'] == 'combo' else ''
        input_vars[config['key']].set(default_value)

    reset_result_panel()


def load_sample_values():
    for config in FIELD_DEFINITIONS:
        input_vars[config['key']].set(SAMPLE_VALUES[config['key']])

    reset_result_panel()


def collect_feature_values():
    features = []

    for config in FIELD_DEFINITIONS:
        raw_value = input_vars[config['key']].get().strip()

        if config['kind'] == 'combo':
            mapped_value = choice_maps[config['key']].get(raw_value)
            if mapped_value is None:
                raise ValueError(f"Please choose {config['label'].lower()}.")
            features.append(mapped_value)
            continue

        if not raw_value:
            raise ValueError(f"Please enter {config['label'].lower()}.")

        try:
            features.append(config['parser'](raw_value))
        except ValueError as error:
            raise ValueError(f"Invalid value for {config['label'].lower()}.") from error

    return features


def predict_heart_disease():
    try:
        features = collect_feature_values()
    except ValueError as error:
        messagebox.showerror("Input Error", str(error))
        return

    features_array = np.array([features])
    features_scaled = scaler.transform(features_array)

    prediction = int(random_model.predict(features_scaled)[0])
    heart_disease_probability = float(random_model.predict_proba(features_scaled)[0][1])
    confidence = heart_disease_probability if prediction == 1 else 1 - heart_disease_probability

    risk_meter['value'] = heart_disease_probability * 100

    if prediction == 1:
        risk_meter.configure(style='RiskHigh.Horizontal.TProgressbar')
        set_result_panel_state(
            'Heart disease likely',
            'The model detected a higher-risk pattern in the submitted profile. Review the inputs and follow up with clinical assessment.',
            f'Estimated heart disease probability: {heart_disease_probability:.1%}',
            f'Model confidence: {confidence:.1%}',
            WARNING,
            WARNING_SOFT,
            'Higher risk'
        )
    else:
        risk_meter.configure(style='RiskLow.Horizontal.TProgressbar')
        set_result_panel_state(
            'No heart disease likely',
            'The submitted values look closer to the lower-risk class learned by the model. This remains a screening estimate, not a diagnosis.',
            f'Estimated heart disease probability: {heart_disease_probability:.1%}',
            f'Model confidence: {confidence:.1%}',
            SUCCESS,
            SUCCESS_SOFT,
            'Lower risk'
        )


def show_analysis_plots():
    try:
        run_full_analysis()
    except Exception as error:
        messagebox.showerror("Analysis Error", f"Could not generate the analysis plots.\n\n{error}")


def update_form_scroll_region(_event=None):
    form_canvas.configure(scrollregion=form_canvas.bbox('all'))


def on_form_mousewheel(event):
    if event.delta:
        form_canvas.yview_scroll(int(-event.delta / 120), 'units')
    elif getattr(event, 'num', None) == 4:
        form_canvas.yview_scroll(-1, 'units')
    elif getattr(event, 'num', None) == 5:
        form_canvas.yview_scroll(1, 'units')


def bind_form_mousewheel(_event=None):
    form_canvas.bind_all('<MouseWheel>', on_form_mousewheel)
    form_canvas.bind_all('<Button-4>', on_form_mousewheel)
    form_canvas.bind_all('<Button-5>', on_form_mousewheel)


def unbind_form_mousewheel(_event=None):
    form_canvas.unbind_all('<MouseWheel>')
    form_canvas.unbind_all('<Button-4>')
    form_canvas.unbind_all('<Button-5>')


# Create the main window
window = tk.Tk()
window.title("Heart Disease Prediction")
window.geometry("1180x760")
window.minsize(1040, 720)
window.configure(bg=APP_BG)
window.lift()

style = ttk.Style(window)
if 'clam' in style.theme_names():
    style.theme_use('clam')

style.configure(
    'Input.TCombobox',
    fieldbackground=FIELD_BG,
    background=FIELD_BG,
    foreground=TEXT_PRIMARY,
    arrowcolor=TEXT_SECONDARY,
    bordercolor=FIELD_BORDER,
    lightcolor=FIELD_BORDER,
    darkcolor=FIELD_BORDER,
    insertcolor=TEXT_PRIMARY
)
style.map(
    'Input.TCombobox',
    fieldbackground=[('readonly', FIELD_BG)],
    selectbackground=[('readonly', FIELD_BG)],
    selectforeground=[('readonly', TEXT_PRIMARY)],
    foreground=[('readonly', TEXT_PRIMARY)]
)
style.configure(
    'Neutral.Horizontal.TProgressbar',
    troughcolor=FIELD_BG,
    bordercolor=FIELD_BG,
    background=GOLD,
    lightcolor=GOLD,
    darkcolor=GOLD
)
style.configure(
    'RiskLow.Horizontal.TProgressbar',
    troughcolor=FIELD_BG,
    bordercolor=FIELD_BG,
    background=SUCCESS,
    lightcolor=SUCCESS,
    darkcolor=SUCCESS
)
style.configure(
    'RiskHigh.Horizontal.TProgressbar',
    troughcolor=FIELD_BG,
    bordercolor=FIELD_BG,
    background=WARNING,
    lightcolor=WARNING,
    darkcolor=WARNING
)

shell = tk.Frame(window, bg=APP_BG)
shell.pack(fill='both', expand=True)
shell.grid_columnconfigure(0, weight=1)
shell.grid_rowconfigure(1, weight=1)

hero_frame = tk.Frame(shell, bg=APP_BG)
hero_frame.grid(row=0, column=0, sticky='ew', padx=36, pady=(28, 18))

hero_left = tk.Frame(hero_frame, bg=APP_BG)
hero_left.pack(side='left', anchor='w')

eyebrow_label = tk.Label(
    hero_left,
    text='Machine Learning Screening Tool',
    bg=APP_BG,
    fg=GOLD,
    font=('Avenir Next', 10, 'bold')
)
eyebrow_label.pack(anchor='w')

hero_title = tk.Label(
    hero_left,
    text='Heart Disease Risk Predictor',
    bg=APP_BG,
    fg=TEXT_PRIMARY,
    font=('Avenir Next', 28, 'bold')
)
hero_title.pack(anchor='w', pady=(6, 4))

hero_subtitle = tk.Label(
    hero_left,
    text='A cleaner clinical-style interface for quick patient screening with the trained Random Forest model.',
    bg=APP_BG,
    fg=TEXT_SECONDARY,
    font=('Avenir Next', 12),
    wraplength=620,
    justify='left'
)
hero_subtitle.pack(anchor='w')

model_badge = tk.Label(
    hero_frame,
    text='Random Forest Model Ready',
    bg='#20364f',
    fg=TEXT_PRIMARY,
    font=('Avenir Next', 11, 'bold'),
    padx=16,
    pady=10
)
model_badge.pack(side='right', anchor='ne')

content_frame = tk.Frame(shell, bg=APP_BG)
content_frame.grid(row=1, column=0, sticky='nsew', padx=36, pady=(0, 32))
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=2)
content_frame.grid_rowconfigure(0, weight=1)

info_panel = tk.Frame(
    content_frame,
    bg=PANEL_BG,
    padx=24,
    pady=24,
    highlightthickness=1,
    highlightbackground='#1d3249'
)
info_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 12))

form_panel = tk.Frame(
    content_frame,
    bg=SURFACE_BG,
    padx=28,
    pady=28,
    highlightthickness=1,
    highlightbackground='#29435f'
)
form_panel.grid(row=0, column=1, sticky='nsew')

overview_label = tk.Label(
    info_panel,
    text='Overview',
    bg=PANEL_BG,
    fg=GOLD,
    font=('Avenir Next', 10, 'bold')
)
overview_label.pack(anchor='w')

overview_title = tk.Label(
    info_panel,
    text='Designed for faster, cleaner patient entry',
    bg=PANEL_BG,
    fg=TEXT_PRIMARY,
    font=('Avenir Next', 18, 'bold'),
    wraplength=320,
    justify='left'
)
overview_title.pack(anchor='w', pady=(8, 10))

overview_body = tk.Label(
    info_panel,
    text='Use the form on the right for the patient profile. Coded fields are now dropdowns, and the result card keeps the prediction visible without extra popup noise.',
    bg=PANEL_BG,
    fg=TEXT_SECONDARY,
    font=('Avenir Next', 11),
    wraplength=320,
    justify='left'
)
overview_body.pack(anchor='w')

chips_frame = tk.Frame(info_panel, bg=PANEL_BG)
chips_frame.pack(anchor='w', fill='x', pady=(18, 22))

for chip_text in ('11 guided inputs', 'Live risk meter', 'Sample row loader'):
    tk.Label(
        chips_frame,
        text=chip_text,
        bg='#17304a',
        fg=TEXT_PRIMARY,
        font=('Avenir Next', 10, 'bold'),
        padx=12,
        pady=8
    ).pack(side='left', padx=(0, 8))

result_card = tk.Frame(
    info_panel,
    bg='#0b1522',
    padx=18,
    pady=18,
    highlightthickness=1,
    highlightbackground=GOLD
)
result_card.pack(fill='x', pady=(0, 18))

result_badge_label = tk.Label(
    result_card,
    text='Ready',
    bg='#2c3e50',
    fg=GOLD,
    font=('Avenir Next', 10, 'bold'),
    padx=10,
    pady=6
)
result_badge_label.pack(anchor='w')

result_title_var = tk.StringVar()
result_title_label = tk.Label(
    result_card,
    textvariable=result_title_var,
    bg='#0b1522',
    fg=TEXT_PRIMARY,
    font=('Avenir Next', 20, 'bold'),
    wraplength=300,
    justify='left'
)
result_title_label.pack(anchor='w', pady=(16, 8))

result_message_var = tk.StringVar()
result_message_label = tk.Label(
    result_card,
    textvariable=result_message_var,
    bg='#0b1522',
    fg=TEXT_SECONDARY,
    font=('Avenir Next', 11),
    wraplength=300,
    justify='left'
)
result_message_label.pack(anchor='w')

probability_var = tk.StringVar()
probability_label = tk.Label(
    result_card,
    textvariable=probability_var,
    bg='#0b1522',
    fg=TEXT_PRIMARY,
    font=('Avenir Next', 11, 'bold')
)
probability_label.pack(anchor='w', pady=(18, 6))

confidence_var = tk.StringVar()
confidence_label = tk.Label(
    result_card,
    textvariable=confidence_var,
    bg='#0b1522',
    fg=TEXT_SECONDARY,
    font=('Avenir Next', 10)
)
confidence_label.pack(anchor='w')

risk_meter = ttk.Progressbar(
    result_card,
    orient='horizontal',
    mode='determinate',
    maximum=100,
    style='Neutral.Horizontal.TProgressbar'
)
risk_meter.pack(fill='x', pady=(18, 10))

result_footer = tk.Label(
    result_card,
    text='This interface supports screening and demonstration. It does not replace clinical diagnosis.',
    bg='#0b1522',
    fg=TEXT_MUTED,
    font=('Avenir Next', 9),
    wraplength=300,
    justify='left'
)
result_footer.pack(anchor='w')

analysis_button = tk.Button(
    info_panel,
    text='Show Full Analysis Plots',
    command=show_analysis_plots,
    bg='#1e3c58',
    fg=TEXT_PRIMARY,
    activebackground='#295276',
    activeforeground=TEXT_PRIMARY,
    font=('Avenir Next', 12, 'bold'),
    relief='flat',
    bd=0,
    padx=16,
    pady=12,
    cursor='hand2'
)
analysis_button.pack(fill='x')

form_tag = tk.Label(
    form_panel,
    text='Patient Input',
    bg=SURFACE_BG,
    fg=GOLD,
    font=('Avenir Next', 10, 'bold')
)
form_tag.pack(anchor='w')

form_title = tk.Label(
    form_panel,
    text='Enter the clinical profile',
    bg=SURFACE_BG,
    fg=TEXT_PRIMARY,
    font=('Avenir Next', 22, 'bold')
)
form_title.pack(anchor='w', pady=(8, 4))

form_intro = tk.Label(
    form_panel,
    text='Numeric measurements stay as typed inputs. Encoded categories are exposed as readable dropdowns.',
    bg=SURFACE_BG,
    fg=TEXT_SECONDARY,
    font=('Avenir Next', 11),
    wraplength=620,
    justify='left'
)
form_intro.pack(anchor='w', pady=(0, 20))

form_scroll_shell = tk.Frame(form_panel, bg=SURFACE_BG)
form_scroll_shell.pack(fill='both', expand=True)
form_scroll_shell.grid_columnconfigure(0, weight=1)
form_scroll_shell.grid_rowconfigure(0, weight=1)

form_canvas = tk.Canvas(
    form_scroll_shell,
    bg=SURFACE_BG,
    highlightthickness=0,
    bd=0
)
form_canvas.grid(row=0, column=0, sticky='nsew')

form_scrollbar = tk.Scrollbar(
    form_scroll_shell,
    orient='vertical',
    command=form_canvas.yview,
    bg=PANEL_BG,
    activebackground='#20364f',
    troughcolor=FIELD_BG,
    relief='flat',
    bd=0
)
form_scrollbar.grid(row=0, column=1, sticky='ns', padx=(10, 0))
form_canvas.configure(yscrollcommand=form_scrollbar.set)

fields_grid = tk.Frame(form_canvas, bg=SURFACE_BG)
form_canvas_window = form_canvas.create_window((0, 0), window=fields_grid, anchor='nw')
fields_grid.grid_columnconfigure(0, weight=1)
fields_grid.grid_columnconfigure(1, weight=1)

for index, config in enumerate(FIELD_DEFINITIONS):
    row = index // 2
    column = index % 2
    is_last_odd_field = index == len(FIELD_DEFINITIONS) - 1 and len(FIELD_DEFINITIONS) % 2 == 1
    columnspan = 2 if is_last_odd_field else 1
    if is_last_odd_field:
        column = 0

    field_card = tk.Frame(
        fields_grid,
        bg=PANEL_BG,
        padx=14,
        pady=14,
        highlightthickness=1,
        highlightbackground='#20364f'
    )
    field_card.grid(row=row, column=column, columnspan=columnspan, padx=8, pady=8, sticky='nsew')

    tk.Label(
        field_card,
        text=config['label'],
        bg=PANEL_BG,
        fg=TEXT_PRIMARY,
        font=('Avenir Next', 11, 'bold')
    ).pack(anchor='w')

    tk.Label(
        field_card,
        text=config['hint'],
        bg=PANEL_BG,
        fg=TEXT_MUTED,
        font=('Avenir Next', 9),
        wraplength=260 if columnspan == 1 else 560,
        justify='left'
    ).pack(anchor='w', pady=(4, 10))

    input_vars[config['key']] = tk.StringVar()

    if config['kind'] == 'combo':
        choice_maps[config['key']] = {label: value for label, value in config['choices']}
        widget = ttk.Combobox(
            field_card,
            textvariable=input_vars[config['key']],
            values=[EMPTY_SELECT] + [label for label, _ in config['choices']],
            state='readonly',
            style='Input.TCombobox',
            font=('Avenir Next', 11)
        )
        widget.set(EMPTY_SELECT)
    else:
        widget = tk.Entry(
            field_card,
            textvariable=input_vars[config['key']],
            bg=FIELD_BG,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief='flat',
            bd=0,
            font=('Avenir Next', 12),
            highlightthickness=1,
            highlightbackground=FIELD_BORDER,
            highlightcolor=ACCENT
        )

    widget.pack(fill='x', ipady=8)
    input_widgets[config['key']] = widget


def resize_form_canvas(event):
    form_canvas.itemconfigure(form_canvas_window, width=event.width)


fields_grid.bind('<Configure>', update_form_scroll_region)
form_canvas.bind('<Configure>', resize_form_canvas)
form_canvas.bind('<Enter>', bind_form_mousewheel)
form_canvas.bind('<Leave>', unbind_form_mousewheel)

button_frame = tk.Frame(form_panel, bg=SURFACE_BG)
button_frame.pack(fill='x', pady=(20, 0))
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

predict_button = tk.Button(
    button_frame,
    text='Predict Risk',
    command=predict_heart_disease,
    bg=ACCENT,
    fg=BUTTON_TEXT_DARK,
    activebackground='#ff9f7a',
    activeforeground=BUTTON_TEXT_DARK,
    font=('Avenir Next', 13, 'bold'),
    relief='flat',
    bd=0,
    padx=16,
    pady=14,
    cursor='hand2'
)
predict_button.grid(row=0, column=0, columnspan=2, sticky='ew')

sample_button = tk.Button(
    button_frame,
    text='Load Sample Row',
    command=load_sample_values,
    bg='#17304a',
    fg=TEXT_PRIMARY,
    activebackground='#214262',
    activeforeground=TEXT_PRIMARY,
    font=('Avenir Next', 11, 'bold'),
    relief='flat',
    bd=0,
    padx=16,
    pady=12,
    cursor='hand2'
)
sample_button.grid(row=1, column=0, sticky='ew', pady=(12, 0), padx=(0, 6))

clear_button = tk.Button(
    button_frame,
    text='Clear Form',
    command=clear_form,
    bg='#0d1b2b',
    fg=TEXT_SECONDARY,
    activebackground='#13283d',
    activeforeground=TEXT_PRIMARY,
    font=('Avenir Next', 11, 'bold'),
    relief='flat',
    bd=0,
    padx=16,
    pady=12,
    cursor='hand2'
)
clear_button.grid(row=1, column=1, sticky='ew', pady=(12, 0), padx=(6, 0))

reset_result_panel()
window.bind('<Return>', lambda _event: predict_heart_disease())

# Start the GUI loop
window.mainloop()

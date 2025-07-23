import os
import joblib
import pandas as pd
import numpy as np
import re
from flask import Flask, request, render_template, jsonify
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import logging
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_feature_names=None):
        self.numerical_feature_names = numerical_feature_names

    def fit(self, X, y=None):
        if self.numerical_feature_names is None and isinstance(X, pd.DataFrame):
            self.numerical_feature_names = X.columns.tolist()
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            if self.numerical_feature_names is None:
                raise ValueError("Feature names must be provided to FeatureEngineer when input is a NumPy array.")
            X_transformed = pd.DataFrame(X, columns=self.numerical_feature_names)
        else:
            X_transformed = X.copy()
        
        if 'time_spent_alone' in X_transformed.columns and 'social_event_attendance' in X_transformed.columns:
            X_transformed['time_social_interaction'] = X_transformed['time_spent_alone'] * X_transformed['social_event_attendance']
            X_transformed['social_alone_ratio'] = X_transformed['social_event_attendance'] / (X_transformed['time_spent_alone'] + 1e-8)

        if 'going_outside' in X_transformed.columns and 'friends_circle_size' in X_transformed.columns:
            X_transformed['outside_friends_interaction'] = X_transformed['going_outside'] * X_transformed['friends_circle_size']

        if 'post_frequency' in X_transformed.columns and 'friends_circle_size' in X_transformed.columns:
            X_transformed['post_friends_interaction'] = X_transformed['post_frequency'] * X_transformed['friends_circle_size']
            X_transformed['friend_post_ratio'] = X_transformed['friends_circle_size'] / (X_transformed['post_frequency'] + 1e-8)
        
        return X_transformed.values

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_feature_names=None):
        self.target_maps = {}
        self.categorical_feature_names = categorical_feature_names

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            if self.categorical_feature_names is None:
                raise ValueError("Categorical feature names must be provided to CustomTargetEncoder when input is a NumPy array.")
            X_df = pd.DataFrame(X, columns=self.categorical_feature_names)
        else:
            X_df = X.copy()

        for col in X_df.columns:
            target_mean = y.groupby(X_df[col].fillna('MissingCategory').astype(str)).mean()
            self.target_maps[col] = target_mean
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.categorical_feature_names is None:
                raise ValueError("Categorical feature names must be provided to CustomTargetEncoder when input is a NumPy array.")
            X_df = pd.DataFrame(X, columns=self.categorical_feature_names)
        else:
            X_df = X.copy()

        X_transformed_df = X_df.copy()
        for col in X_transformed_df.columns:
            if col in self.target_maps:
                X_transformed_df[col] = X_transformed_df[col].fillna('MissingCategory').astype(str).map(self.target_maps[col]).fillna(0.5)
            else:
                X_transformed_df[col] = X_transformed_df[col].fillna(0.5)
        return X_transformed_df.values

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators=None, voting='soft', weights=None):
        self.estimators = estimators if estimators is not None else [
            ('lgbm', LGBMClassifier(random_state=42)),
            ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
            ('cat', CatBoostClassifier(random_state=42, verbose=0))
        ]
        self.voting = voting
        self.weights = weights
        self.ensemble_classifier = VotingClassifier(
            estimators=self.estimators,
            voting=self.voting,
            weights=self.weights
        )

    def fit(self, X, y):
        self.ensemble_classifier.fit(X, y)
        self.classes_ = self.ensemble_classifier.classes_
        return self

    def predict(self, X):
        return self.ensemble_classifier.predict(X)

    def predict_proba(self, X):
        return self.ensemble_classifier.predict_proba(X)

MODEL_PATH = os.path.join('saved_models', 'personality_predictor_artifacts.pkl')
final_pipeline = None
target_label_map = None

if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

try:
    artifacts = joblib.load(MODEL_PATH)
    final_pipeline = artifacts['final_pipeline']
    target_label_map = artifacts['target_label_map']
    logger.info(f"Model artifacts loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Error: Model artifacts file not found at {MODEL_PATH}. Please ensure it's in the correct directory.")
    logger.error("You need to run the model training and save the 'personality_predictor_artifacts.pkl' from your Jupyter Notebook.")
    final_pipeline = None
    target_label_map = None
except Exception as e:
    logger.error(f"Error loading model artifacts: {e}. This often means a custom class (like FeatureEngineer or EnsembleModel) is not defined or is defined differently than when the model was saved, or there's a library version mismatch (e.g., numpy, scikit-learn).", exc_info=True)
    final_pipeline = None
    target_label_map = None

def clean_column_names_for_input(df):
    new_columns = []
    for col in df.columns:
        new_col = col.strip().replace(' ', '_').lower()
        new_col = re.sub(r'[^a-z0-9_]', '', new_col)
        new_columns.append(new_col)
    df.columns = new_columns
    return df

def preprocess_input(input_data_dict):
    df_input = pd.DataFrame([input_data_dict])
    df_input = clean_column_names_for_input(df_input)

    expected_cols_after_clean = [
        'time_spent_alone',
        'social_event_attendance',
        'going_outside',
        'friends_circle_size',
        'post_frequency',
        'stage_fear',
        'drained_after_socializing'
    ]

    for col in expected_cols_after_clean:
        if col not in df_input.columns:
            if col in ['time_spent_alone', 'social_event_attendance', 'going_outside', 'friends_circle_size', 'post_frequency']:
                df_input[col] = np.nan
            else:
                df_input[col] = None
            
    df_input = df_input[expected_cols_after_clean]

    return df_input

def create_confidence_plot(proba_values, target_label_map):
    plt.figure(figsize=(8, 4))
    
    labels = [target_label_map.get(0, 'Class 0'), target_label_map.get(1, 'Class 1')]
    colors = ['#e76f51', '#2a9d8f']

    plt.barh(labels, [proba_values[0], proba_values[1]], color=colors)
    plt.title('Prediction Confidence', fontsize=16)
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return plot_data

@app.route('/')
def home():
    return render_template('index.html', prediction_result=None, confidence=None, input_data={}, error=None, plot_url=None)

@app.route('/predict', methods=['POST'])
def predict():
    data = {}
    input_data_for_template = {}

    try:
        data = {
            'Time spent alone': float(request.form['time_spent_alone']),
            'Social event attendance': int(request.form['social_event_attendance']),
            'Going outside': int(request.form['going_outside']),
            'Friends circle size': int(request.form['friends_circle_size']),
            'Post frequency': int(request.form['post_frequency']),
            'Stage fear': request.form['stage_fear'],
            'Drained after socializing': request.form['drained_after_socializing']
        }
        input_data_for_template = {
            k.strip().replace(' ', '_').lower(): v 
            for k, v in data.items()
        }

        if final_pipeline is None or target_label_map is None:
            error_msg = "Error: Model components not loaded. Please check server logs and ensure the model is trained and saved."
            logger.error(error_msg)
            return render_template('index.html',
                                   prediction_result=None,
                                   confidence=None,
                                   plot_url=None,
                                   error=error_msg,
                                   input_data=input_data_for_template)

        processed_df = preprocess_input(data)

        probabilities = final_pipeline.predict_proba(processed_df)[0]
        prediction_encoded = final_pipeline.predict(processed_df)[0]
        
        predicted_label = target_label_map.get(prediction_encoded, 'Unknown')
        
        confidence = probabilities[prediction_encoded]
        plot_url = create_confidence_plot(probabilities, target_label_map)

        return render_template('index.html',
                               prediction_result=predicted_label,
                               confidence=confidence,
                               plot_url=plot_url,
                               input_data=input_data_for_template,
                               error=None)

    except ValueError as ve:
        error_msg = f"Input Error: Please ensure all fields are correctly filled and are numbers where expected. Details: {ve}"
        logger.error(error_msg, exc_info=True)
        return render_template('index.html',
                               prediction_result=None,
                               confidence=None,
                               plot_url=None,
                               error=error_msg,
                               input_data=input_data_for_template)
    except KeyError as ke:
        error_msg = f"Missing Input: A required field was not provided. Details: {ke}"
        logger.error(error_msg, exc_info=True)
        return render_template('index.html',
                               prediction_result=None,
                               confidence=None,
                               plot_url=None,
                               error=error_msg,
                               input_data=input_data_for_template)
    except Exception as e:
        error_msg = f"An unexpected error occurred during prediction: {e}"
        logger.error(error_msg, exc_info=True)
        return render_template('index.html',
                               prediction_result=None,
                               confidence=None,
                               plot_url=None,
                               error=error_msg,
                               input_data=input_data_for_template)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    if final_pipeline is None or target_label_map is None:
        return jsonify({"error": "Model components not loaded on server."}), 500

    try:
        api_to_form_map = {
            'time_spent_alone': 'Time spent alone',
            'social_event_attendance': 'Social event attendance',
            'going_outside': 'Going outside',
            'friends_circle_size': 'Friends circle size',
            'post_frequency': 'Post frequency',
            'stage_fear': 'Stage fear',
            'drained_after_socializing': 'Drained after socializing'
        }
        form_data_from_api = {api_to_form_map.get(k, k): v for k, v in data.items()}

        # Directly use final_pipeline for prediction
        processed_df = preprocess_input(form_data_from_api)
        personality_encoded = final_pipeline.predict(processed_df)[0]
        probabilities = final_pipeline.predict_proba(processed_df)[0]

        personality = target_label_map.get(personality_encoded, 'Unknown')
        confidence = probabilities[personality_encoded]

        return jsonify({
            "personality": personality,
            "confidence": confidence,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    app.run(host='0.0.0.0', port=5000, debug=True)

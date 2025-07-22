from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

artifacts = joblib.load('model/personality_classifier.pkl')
feature_engineer = artifacts['feature_engineer']
preprocessor = artifacts['preprocessor']
ensemble = artifacts['ensemble']
feature_names = artifacts['feature_names']
best_threshold = artifacts.get('best_threshold', 0.5)

def predict_personality(input_data):
    clean_input = {}
    for key, value in input_data.items():
        clean_key = re.sub(r'[^a-z0-9]', '', key.lower().replace(' ', '_'))
        clean_input[clean_key] = value
    
    input_df = pd.DataFrame([clean_input])
    
    input_fe = feature_engineer.transform(input_df)
    
    input_preprocessed = preprocessor.transform(input_fe)
    
    proba = ensemble.predict_proba(input_preprocessed)[0]
    prediction = ensemble.predict(input_preprocessed, threshold=best_threshold)[0]
    
    personality = "extrovert" if prediction == 1 else "introvert"
    confidence = proba[1] if personality == "extrovert" else proba[0]
    
    return personality, confidence, proba

def create_confidence_plot(proba):
    """Create a confidence plot as base64 encoded image"""
    plt.figure(figsize=(8, 4))
    plt.barh(['Introvert', 'Extrovert'], [proba[0], proba[1]], 
             color=['#2a9d8f', '#e76f51'])
    plt.title('Prediction Confidence', fontsize=16)
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return plot_data

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        
        numerical_features = [f for f in feature_names if f not in ['stage_fear', 'drained_after_socializing']]
        for feat in numerical_features:
            if feat in form_data:
                try:
                    form_data[feat] = float(form_data[feat])
                except:
                    form_data[feat] = 0.0
        
        try:
            personality, confidence, proba = predict_personality(form_data)
            plot_url = create_confidence_plot(proba)
            return render_template('result.html', 
                                  personality=personality,
                                  confidence=f"{confidence*100:.1f}%",
                                  plot_url=plot_url,
                                  inputs=form_data)
        except Exception as e:
            return render_template('index.html', 
                                  error=f"Prediction error: {str(e)}", 
                                  features=feature_names)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    try:
        personality, confidence, _ = predict_personality(data)
        return jsonify({
            "personality": personality,
            "confidence": confidence,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
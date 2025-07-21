from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('saved_models/best_model.pkl')
preprocessor = joblib.load('saved_models/best_preprocessor.pkl')

# Top features for simplified form
TOP_FEATURES = [
    'social_going', 'friend_post', 'avoids_interaction',
    'drained_going', 'alone', 'social_engagement'
]

@app.route('/')
def index():
    return render_template('index.html', features=TOP_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = {}
    for feature in TOP_FEATURES:
        form_data[feature] = float(request.form[feature])
    
    # Create a DataFrame with all features
    features_df = pd.DataFrame([form_data])
    
    # Add missing features with default values
    all_features = [
        'age', 'social_going', 'friend_post', 'avoids_interaction', 
        'drained_going', 'alone', 'social_engagement', 'social_std',
        'friend_post_ratio', 'friend_post_product', 'social_index',
        'drained_going_interaction', 'alone_ratio', 'alone_log',
        'behavioral_consistency', 'behavioral_variance', 'alone_social_interaction'
    ]
    
    for feature in all_features:
        if feature not in features_df.columns:
            # Set reasonable defaults
            if 'ratio' in feature or 'log' in feature:
                features_df[feature] = 0.5
            else:
                features_df[feature] = np.random.randint(1, 5)
    
    # Preprocess data
    processed_data = preprocessor.transform(features_df)
    
    # Make prediction
    prediction = model.predict(processed_data)[0]
    personality = "Extrovert" if prediction == 1 else "Introvert"
    
    # Get prediction probability
    proba = model.predict_proba(processed_data)[0]
    confidence = max(proba) * 100
    
    return render_template('result.html', 
                          personality=personality, 
                          confidence=f"{confidence:.1f}%",
                          features=form_data)

if __name__ == '__main__':
    app.run(debug=True)
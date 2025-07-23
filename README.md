# Personality Predictor Web Application

This is a Flask-based web application that uses a pre-trained machine learning model to predict a user's personality type (Introvert or Extrovert) based on several input features.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Prerequisites](#prerequisites)
4.  [Project Structure](#project-structure)
5.  [Setup and Installation](#setup-and-installation)
6.  [Running the Application](#running-the-application)
7.  [Model Details](#model-details)
8.  [API Endpoint](#api-endpoint)
9.  [Contributing](#contributing)
10. [License](#license)

## 1. Introduction

This web application provides a user-friendly interface to interact with a personality prediction model. Users can input various personal and social metrics through a web form, and the application will return a prediction of whether they are more likely to be an "Introvert" or "Extrovert," along with a confidence score.

## 2. Features

* **Web-based User Interface:** Simple and intuitive form for inputting data.
* **Real-time Predictions:** Get instant personality predictions.
* **Confidence Score:** Understand the model's certainty in its prediction.
* **Input Data Display:** Review the data you submitted for prediction.
* **RESTful API Endpoint:** Programmatic access to the prediction service.

## 3. Prerequisites

Before running this application, ensure you have the following installed:

* Python 3.8+
* `pip` (Python package installer)

## 4. Project Structure

```
personality-app/
├── app.py
├── saved_models/
│   └── personality_predictor_artifacts.pkl
├── templates/
│   └── index.html
└── static/
    └── style.css
```

* `app.py`: The main Flask application file containing the backend logic, model loading, and prediction endpoints.
* `saved_models/`: Directory to store the trained machine learning model artifacts.
    * `personality_predictor_artifacts.pkl`: The serialized (pickled) machine learning pipeline, including preprocessing steps and the trained ensemble model, along with the target label map.
* `templates/`: Directory for HTML template files.
    * `index.html`: The single HTML file containing both the input form and the prediction result display.
* `static/`: Directory for static assets like CSS.
    * `style.css`: Custom CSS for styling the web application.

## 5. Setup and Installation

1.  **Clone the repository (if applicable) or create the project directory:**
    ```bash
    mkdir personality-app
    cd personality-app
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required Python packages:**
    ```bash
    pip install Flask pandas numpy scikit-learn lightgbm xgboost catboost joblib matplotlib
    ```
    *Note: You might see `InconsistentVersionWarning` if your `scikit-learn` version used for saving the model is different from the one installed. It's highly recommended to retrain and save your model using the same `scikit-learn` version as your deployment environment to avoid potential issues.*

5.  **Place the trained model:**
    Ensure you have run your Jupyter notebook (`introvert-extrovert-pred.ipynb` or the code from `introvert-extrovert-prediction-python-code-fixed` Canvas) to train and save the model. The model artifact file `personality_predictor_artifacts.pkl` should be placed inside the `saved_models/` directory. Create this directory if it doesn't exist.

## 6. Running the Application

1.  **Navigate to the project root directory** (where `app.py` is located).
2.  **Ensure your virtual environment is activated.**
3.  **Run the Flask application:**
    ```bash
    python app.py
    ```
4.  **Open your web browser** and go to `http://127.0.0.1:5000/` (or `http://localhost:5000/`).

## 7. Model Details

The core of this application is a machine learning model trained to classify individuals as "Introvert" or "Extrovert."

**Input Features (from the web form):**

* `Time spent alone` (numerical, hours/day)
* `Social event attendance` (numerical, times/month)
* `Going outside` (numerical, times/week)
* `Friends circle size` (numerical)
* `Post frequency` (numerical, posts/month)
* `Stage fear` (categorical: 'Yes', 'No')
* `Drained after socializing` (categorical: 'Yes', 'No')

**Output:**

* **Predicted Personality:** "Introvert" or "Extrovert"
* **Confidence:** A probability score (0.00% - 100.00%) indicating the model's confidence in the predicted class.
* **Confidence Plot:** A horizontal bar chart visualizing the probabilities for both classes.

The model itself is an **Ensemble Model** (specifically a `VotingClassifier` with `soft` voting) composed of:
* LightGBM Classifier (`LGBMClassifier`)
* XGBoost Classifier (`XGBClassifier`)
* CatBoost Classifier (`CatBoostClassifier`)

The preprocessing pipeline includes:
* `SimpleImputer` for handling missing values (mean for numerical, most frequent for categorical).
* `StandardScaler` for scaling numerical features.
* `FeatureEngineer` for creating interaction terms and polynomial features.
* `OrdinalEncoder` for encoding categorical features.

## 8. API Endpoint

You can also interact with the prediction service programmatically via a RESTful API endpoint.

* **Endpoint:** `/api/predict`
* **Method:** `POST`
* **Content-Type:** `application/json`

**Request Body Example:**

```json
{
    "time_spent_alone": 8.5,
    "social_event_attendance": 1,
    "going_outside": 2,
    "friends_circle_size": 5,
    "post_frequency": 1,
    "stage_fear": "Yes",
    "drained_after_socializing": "Yes"
}
```

**Response Body Example (Success):**

```json
{
    "confidence": 0.85,
    "personality": "Introvert",
    "status": "success"
}
```

**Response Body Example (Error):**

```json
{
    "error": "Input Error: Please ensure all fields are correctly filled and are numbers where expected. Details: invalid literal for float(): abc"
}
```

## 9. Contributing

Feel free to fork this repository, submit pull requests, or open issues for any improvements or bug fixes.

## 10. License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
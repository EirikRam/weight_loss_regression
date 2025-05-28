from flask import Flask, render_template, request
import numpy as np
from src.data_preprocessing import load_and_preprocess_data
from keras.models import load_model
import joblib


# Helper for better placeholders
def format_input(name):
    placeholder_map = {
        'Gender': '1 for male, 0 for female',
        'Age': 'Years (e.g., 30)',
        'Height': 'in cm (e.g., 175)',
        'Weight': 'in kg (e.g., 75)',
        'Duration': 'in minutes (e.g., 60)',
        'Heart_Rate': 'e.g., 140',
        'Body_Temp': 'in °C (e.g., 37)',
    }
    return placeholder_map.get(name, f'{name}')

# Initialize Flask app
app = Flask(__name__)
app.jinja_env.filters['format_input'] = format_input


# Load and preprocess data to get feature order and scaler context
X_train, X_test, y_train, y_test, feature_names, scalar = load_and_preprocess_data()

# Load pre-trained and regularized model (improves consistency and avoids retraining on each app launch)
model = load_model('models/calorie_model.h5')

# Load the scaler in the Flask app
scaler = joblib.load('models/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        input_data = []
        for feature in feature_names:
            val = float(request.form.get(feature))
            input_data.append(val)

        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        pred = model.predict(scaled_input)
        prediction = round(pred[0][0], 2)

    return render_template('index.html', feature_names=feature_names, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

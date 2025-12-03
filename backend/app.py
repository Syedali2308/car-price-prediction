import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS 
import pandas as pd
import numpy as np


# --- 1. Load your Model and Preprocessor ---
try:
    # IMPORTANT: Ensure your model file is named 'car_price_predictor.pkl' 
    # and is in the same directory.
    model = joblib.load('carprediction.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 2. Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- 3. Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    try:
        data = request.get_json(silent=True)
        if not data:
             return jsonify({'error': 'Invalid JSON data received.'}), 400

        # === FIX APPLIED HERE ===
        # If 'data' is a list (array), extract the first item (the dictionary)
        if isinstance(data, list):
            # Check if the list is empty before trying to access index 0
            if data:
                data = data[0]
            else:
                return jsonify({'error': 'Received an empty list of data.'}), 400
        # ========================

        # Create a Pandas DataFrame from the input data
        # Ensure the column names and order match EXACTLY what your model expects
        input_data = pd.DataFrame([{
            'Company': data.get('Company'),
            'year': int(data.get('year')),
            'km_driven': int(data.get('km_driven')),
            'fuel': data.get('fuel'),
            'transmission': data.get('transmission'),
            'owner': data.get('owner'),
            'seller_type': data.get('seller_type'),
            'seats': int(data.get('seats')),
            # Add any calculated features your model needs (e.g., Car_Age)
        }])
        
        # --- 4. Predict ---
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'predicted_price': np.round(prediction, 2)
        })

    except Exception as e:
        # Log the error for debugging
        print(f"Prediction failed due to: {e}")
        # Return a clean error message to the frontend
        return jsonify({'error': f'Prediction processing failed on server (check terminal for details).'}), 500

# --- 5. Run the Server ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
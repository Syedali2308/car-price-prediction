import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import traceback

# Load your trained pipeline
try:
    model = joblib.load('carprediction.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'Invalid JSON data received.'}), 400

        # Handle list input
        if isinstance(data, list) and data:
            data = data[0]

        # Validate required fields
        required_fields = ['Company','year','km_driven','fuel','transmission','owner','seller_type','seats']
        for field in required_fields:
            if field not in data or data[field] in [None, ""]:
                return jsonify({'error': f'Missing or empty field: {field}'}), 400

        # Capitalize company to reduce unknown category issues
        company = str(data.get('Company')).strip()
        company = company[0].upper() + company[1:].lower()

        # Prepare input DataFrame
        input_data = pd.DataFrame([{
            'Company': company,
            'year': int(data.get('year')),
            'km_driven': int(data.get('km_driven')),
            'fuel': data.get('fuel'),
            'transmission': data.get('transmission'),
            'owner': data.get('owner'),
            'seller_type': data.get('seller_type'),
            'seats': int(data.get('seats')),
        }])

        # Predict
        prediction = model.predict(input_data)[0]

        return jsonify({'predicted_price': float(np.round(prediction, 2))})

    except Exception as e:
        print("=== PREDICTION ERROR ===")
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed on server. Check server logs.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

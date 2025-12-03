import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

# Load model
try:
    model = joblib.load('carprediction.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize Flask app
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

        if isinstance(data, list):
            if data:
                data = data[0]
            else:
                return jsonify({'error': 'Received empty data list.'}), 400

        # Build input DataFrame
        input_data = pd.DataFrame([{
            'Company': data.get('Company'),
            'year': int(data.get('year')),
            'km_driven': int(data.get('km_driven')),
            'fuel': data.get('fuel'),
            'transmission': data.get('transmission'),
            'owner': data.get('owner'),
            'seller_type': data.get('seller_type'),
            'seats': int(data.get('seats'))
        }])

        prediction = model.predict(input_data)[0]
        return jsonify({'predicted_price': np.round(prediction, 2)})

    except Exception as e:
        print(f"Prediction failed due to: {e}")
        return jsonify({'error': 'Prediction failed on server. Check logs.'}), 500

if __name__ == "__main__":
    # Detect if running on Render or locally
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

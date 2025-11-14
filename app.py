from flask import Flask, request, jsonify
import joblib, numpy as np
from datetime import datetime
import os

app = Flask(__name__)
# Load the pre-trained model
model = joblib.load("iris_model.pkl")
LOG_FILE = "/var/log/ml_api.log"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # The expected input is a list of features, e.g., '{"features":[5.1,3.5,1.4,0.2]}'
    features = np.array(data['features']).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(features)[0]
    
    # Create log entry
    log = {
        'timestamp': datetime.utcnow().isoformat(),
        'input': data['features'],
        'prediction': str(prediction)
    }
    
    # Log the request details
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(str(log) + "\n")
    except Exception as e:
        print(f"Could not write to log file: {e}")

    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    # Run the app, accessible from any IP on port 5000
    app.run(host='0.0.0.0', port=5000)

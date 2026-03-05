import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            'CPU': data['CPU'],
            'GPU': data['GPU'],
            'RAM': data['RAM'],
            'Storage': data['Storage'],
            'Rating': float(data['Rating'])
        }])

        prediction_log = model.predict(input_df)
        prediction = np.expm1(prediction_log)[0]

        return jsonify({'prediction': round(float(prediction), 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug = True)
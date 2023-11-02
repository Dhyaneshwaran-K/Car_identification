# app.py

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load your H5 model
model = keras.models.load_model('model_resnet50.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data['input']
    
    # Perform model prediction
    predictions = model.predict(input_data)
    
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

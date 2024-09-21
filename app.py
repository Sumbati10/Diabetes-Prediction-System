from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the pickled model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json

    # Prepare the input data for prediction
    input_data = np.array([[data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                            data['SkinThickness'], data['Insulin'], data['BMI'],
                            data['DiabetesPedigreeFunction'], data['Age']]])

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the prediction
    return jsonify({'Prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

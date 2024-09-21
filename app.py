import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Load dataset (if needed for other purposes)
dataset = pd.read_csv('diabetes.csv')
dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    pregnancies = float(request.form['Pregnancies'])
    glucose = float(request.form['Glucose'])
    blood_pressure = float(request.form['BloodPressure'])
    skin_thickness = float(request.form['SkinThickness'])
    insulin = float(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
    age = float(request.form['Age'])

    # Create a feature array with all 8 features
    final_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, diabetes_pedigree_function, age]])

    # Make prediction without scaling
    prediction = model.predict(final_features)

    # Determine output message based on prediction
    if prediction == 1:
        output = "You have Diabetes, please consult a Doctor."
    else:
        output = "You don't have Diabetes."

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

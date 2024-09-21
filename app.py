import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the logistic regression model
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Load the dataset
dataset = pd.read_csv('diabetes.csv')
dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values  # Adjusted to match relevant columns

# Scale the dataset
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    # Scale the input features
    scaled_features = sc.transform(final_features)
    
    # Make prediction
    prediction = model.predict(scaled_features)

    # Prepare output message
    if prediction == 1:
        output = "You have Diabetes, please consult a Doctor."
    else:
        output = "You don't have Diabetes."

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

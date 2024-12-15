
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\Abhay\OneDrive\Desktop\xyz\iris_model.pk1')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')  

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.form  # Data from the form
    features = np.array([[
        float(data['sepal_length']),
        float(data['sepal_width']),
        float(data['petal_length']),
        float(data['petal_width'])
    ]])

    # Make a prediction
    prediction = model.predict(features)[0]
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    result = species[prediction]

    return render_template('result.html',prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

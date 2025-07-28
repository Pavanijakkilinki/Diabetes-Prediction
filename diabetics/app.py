from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('svm_model.pkl', 'rb') as model_file:
        classifier = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    classifier, scaler = None, None

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = [float(request.form[feature]) for feature in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]

        # Convert input data to a numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)

        # Standardize the input data
        std_data = scaler.transform(input_array)

        # Predict the result
        prediction = classifier.predict(std_data)[0]

        # Return result
        result_message = "The person is diabetic." if prediction == 1 else "The person is not diabetic."
        return render_template('result.html', prediction=result_message)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")

# Run the app
if __name__== '__main__':
    app.run(debug=True)
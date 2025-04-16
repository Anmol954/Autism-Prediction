from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and encoders
model = pickle.load(open('best_model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Safe encoding to handle unseen labels
def safe_encode(encoder, value):
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, "unknown")
        value = "unknown"
    return encoder.transform([value])[0]

@app.route('/')
def home():
    return render_template('web.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract and clean input values
        age = int(data['age'])
        gender = data['gender'].lower()
        ethnicity = data['ethnicity'].lower()
        jaundice = data['jaundice'].lower()
        autism = data['austim'].lower()
        relation = data['relation'].lower()
        country = data['contry_of_res'].lower()  # Fixed: use the actual encoder key
        used_app = data['used_app_before'].lower()
        test_result = int(data['test_result'])

        # Extract A1 to A10 Scores
        scores = [int(data[f'A{i}_Score']) for i in range(1, 11)]

        # Encode categorical values safely
        gender_encoded = safe_encode(encoders['gender'], gender)
        ethnicity_encoded = safe_encode(encoders['ethnicity'], ethnicity)
        jaundice_encoded = safe_encode(encoders['jaundice'], jaundice)
        autism_encoded = safe_encode(encoders['austim'], autism)
        relation_encoded = safe_encode(encoders['relation'], relation)
        used_app_encoded = safe_encode(encoders['used_app_before'], used_app)
        country_encoded = safe_encode(encoders['contry_of_res'], country)  # Fixed key

        # Combine all input features
        input_data = np.array([
            age, gender_encoded, *scores, ethnicity_encoded, jaundice_encoded,
            autism_encoded, used_app_encoded, test_result, relation_encoded, country_encoded
        ]).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)
        result = 'At Risk' if prediction[0] == 1 else 'Not at Risk'

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

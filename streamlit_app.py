import streamlit as st
import pickle
import numpy as np

# Load the trained model and encoders
model = pickle.load(open('best_model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Safe encoding for unseen labels
def safe_encode(encoder, value):
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, "unknown")
        value = "unknown"
    return encoder.transform([value])[0]

# Streamlit app
st.title("Autism Spectrum Disorder Prediction App")

st.write("Please fill the details below:")

age = st.number_input("Age", min_value=1, max_value=100)

gender = st.selectbox("Gender", ["male", "female"])
ethnicity = st.text_input("Ethnicity", "White-European")
jaundice = st.selectbox("Jaundice at birth", ["yes", "no"])
autism = st.selectbox("Family history of autism", ["yes", "no"])
relation = st.text_input("Relation to patient", "Self")
country = st.text_input("Country of residence", "United States")
used_app_before = st.selectbox("Used screening app before", ["yes", "no"])
test_result = st.number_input("AQ-10 Screening Test Score", min_value=0, max_value=10)

st.write("Enter AQ-10 Scores (0 or 1) for 10 behavioral questions:")
scores = []
for i in range(1, 11):
    score = st.selectbox(f"A{i}_Score", [0, 1], key=f"A{i}")
    scores.append(score)

if st.button("Predict"):
    try:
        # Encode categorical values
        gender_encoded = safe_encode(encoders['gender'], gender)
        ethnicity_encoded = safe_encode(encoders['ethnicity'], ethnicity)
        jaundice_encoded = safe_encode(encoders['jaundice'], jaundice)
        autism_encoded = safe_encode(encoders['austim'], autism)
        relation_encoded = safe_encode(encoders['relation'], relation)
        used_app_encoded = safe_encode(encoders['used_app_before'], used_app_before)
        country_encoded = safe_encode(encoders['contry_of_res'], country)

        # Prepare input
        input_data = np.array([
            age, gender_encoded, *scores, ethnicity_encoded, jaundice_encoded,
            autism_encoded, used_app_encoded, test_result, relation_encoded, country_encoded
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        result = "At Risk" if prediction[0] == 1 else "Not at Risk"

        st.success(f"The prediction is: {result}")

    except Exception as e:
        st.error(f"Error: {e}")

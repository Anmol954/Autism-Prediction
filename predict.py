import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the label encoders
with open("encoders.pkl", "rb") as encoders_file:
    encoders = pickle.load(encoders_file)

# Function to get user input
def get_user_input():
    return pd.DataFrame([{
        "A1_Score": int(input("A1_Score (0 or 1): ")), 
        "A2_Score": int(input("A2_Score (0 or 1): ")), 
        "A3_Score": int(input("A3_Score (0 or 1): ")), 
        "A4_Score": int(input("A4_Score (0 or 1): ")), 
        "A5_Score": int(input("A5_Score (0 or 1): ")), 
        "A6_Score": int(input("A6_Score (0 or 1): ")), 
        "A7_Score": int(input("A7_Score (0 or 1): ")), 
        "A8_Score": int(input("A8_Score (0 or 1): ")), 
        "A9_Score": int(input("A9_Score (0 or 1): ")), 
        "A10_Score": int(input("A10_Score (0 or 1): ")), 
        "age": int(input("Enter age: ")), 
        "gender": input("Gender (male/female): "), 
        "ethnicity": input("Ethnicity: "), 
        "jaundice": input("Jaundice (yes/no): "), 
        "austim": input("Autism history (yes/no): "), 
        "contry_of_res": input("Country of residence: "), 
        "used_app_before": input("Used app before (yes/no): "), 
        "result": float(input("Test result score: ")), 
        "relation": input("Relation (Self/Parent/etc.): ")
    }])

# Get user input
data = get_user_input()

# Encode categorical values using the saved encoders
for col in encoders:
    if col in data.columns:
        # Handle unseen labels
        data[col] = data[col].apply(lambda x: x if x in encoders[col].classes_ else "Unknown")
        
        # Add "Unknown" class if missing in encoder
        if "Unknown" not in encoders[col].classes_:
            encoders[col].classes_ = np.array(list(encoders[col].classes_) + ["Unknown"])
        
        # Transform categorical values
        data[col] = encoders[col].transform(data[col])

# Make a prediction
prediction = model.predict(data)

# Output the result
print("Autism Prediction:", "At Risk" if prediction[0] == 1 else "Not at Risk")

import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the label encoders
with open("encoders.pkl", "rb") as encoders_file:
    encoders = pickle.load(encoders_file)

# Define all features expected by the model
feature_columns = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score", 
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score", 
    "age", "gender", "ethnicity", "jaundice", "austim", 
    "contry_of_res", "used_app_before", "result", "relation"
]

# Example new data (Replace with actual user input)
new_data = pd.DataFrame([{
    "A1_Score": 1, "A2_Score": 0, "A3_Score": 1, "A4_Score": 0, "A5_Score": 1,
    "A6_Score": 0, "A7_Score": 1, "A8_Score": 0, "A9_Score": 1, "A10_Score": 0,
    "age": 25, "gender": "male", "ethnicity": "White-European", "jaundice": "no", 
    "austim": "no", "contry_of_res": "United States", "used_app_before": "no", 
    "result": 6.35, "relation": "Self"
}])

# Encode categorical values using the saved encoders
for col in encoders:
    if col in new_data.columns:
        # Handle unseen labels by assigning "Unknown"
        new_data[col] = new_data[col].apply(lambda x: x if x in encoders[col].classes_ else "Unknown")

        # Add "Unknown" class if missing in encoder
        if "Unknown" not in encoders[col].classes_:
            encoders[col].classes_ = np.array(list(encoders[col].classes_) + ["Unknown"])

        # Transform categorical values
        new_data[col] = encoders[col].transform(new_data[col])

# Ensure correct column order
new_data = new_data[feature_columns]

# Make a prediction
prediction = model.predict(new_data)

# Output the result
print("Autism Prediction:", "At Risk" if prediction[0] == 1 else "Not at Risk")
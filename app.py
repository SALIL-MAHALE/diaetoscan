#pip install -r requirements.txt

import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------

# Load dataset (update the file path as needed)
df = pd.read_csv('C:\\salilpython\\minprojectsem4\\dataset_diab\\diabetes.csv')

# Drop the 'Insulin' column as per the project requirements
df = df.drop('Insulin', axis=1)

# Define features and target.
# Our model uses: Glucose, BloodPressure, SkinThickness, BMI, Age, Pregnancies, DiabetesPedigreeFunction.
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# We scale only the following columns
scale_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age']
# Final order of features our model expects:
features_order = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction']

# Create a ColumnTransformer to scale the selected columns and pass the rest as is.
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), scale_cols)
    ],
    remainder='passthrough'
)

# Apply transformation on the features.
X_transformed = preprocessor.fit_transform(X)
X_prepared = pd.DataFrame(X_transformed, columns=features_order)

# -------------------------------
# 2. Model Training
# -------------------------------

# Train logistic regression on the full dataset.
model = LogisticRegression(random_state=42, n_jobs=-1, class_weight='balanced', max_iter=800, C=0.8)
model.fit(X_prepared, y)

# -------------------------------
# 3. Flask Endpoints
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve input values from the form and convert them to float.
            num_pregnancies = float(request.form['num_pregnancies'])
            glucose_level = float(request.form['glucose_level'])
            blood_pressure = float(request.form['blood_pressure'])
            skin_thickness = float(request.form['skin_thickness'])
            bmi = float(request.form['bmi'])
            diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
            age = float(request.form['age'])
            
            # Build input data dictionary with keys matching the training order.
            input_data = {
                'Glucose': glucose_level,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'BMI': bmi,
                'Age': age,
                'Pregnancies': num_pregnancies,
                'DiabetesPedigreeFunction': diabetes_pedigree_function
            }
            
            # Create a DataFrame with the input data ensuring the correct order.
            input_df = pd.DataFrame([input_data], columns=features_order)
            
            # Preprocess the input using the already fitted preprocessor.
            input_transformed = preprocessor.transform(input_df)
            input_prepared = pd.DataFrame(input_transformed, columns=features_order)
            
            # Get the predicted probability (for class 1: disease present) and class prediction.
            probability = model.predict_proba(input_prepared)[0, 1]
            prediction = model.predict(input_prepared)[0]
            
            # Prepare the result with a rounded probability and a human-readable prediction.
            result = {
                "disease_probability": round(probability, 4),
                "prediction": "Disease present" if prediction == 1 else "No disease"
            }
            
            return render_template('predict.html', result=result)
            
        except Exception as e:
            return render_template('predict.html', result={"error": str(e)})
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)

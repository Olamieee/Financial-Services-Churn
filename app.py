# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import traceback

app = Flask(__name__)

# Load the saved model, encoder, and scaler
model = joblib.load('financial_churn_model.pkl')
encoder = joblib.load('encoder.pkl')  # Assuming this is a dict of LabelEncoders for categorical columns
scaler = joblib.load('scaler.pkl')

# Define categorical columns based on the dataset
cat_cols = ['gender', 'state', 'city_type', 'occupation', 'education_level', 'marital_status', 'telecom_provider']

# Define all feature columns (23 features)
feature_cols = ['gender', 'age', 'state', 'city_type', 'occupation', 'education_level', 'marital_status',
                'num_dependents', 'income_monthly_ngn', 'avg_balance_ngn', 'monthly_deposit_ngn',
                'monthly_withdrawal_ngn', 'monthly_transactions', 'num_products', 'has_mobile_app',
                'uses_agent', 'loan_active', 'savings_goal_met', 'recent_complaints', 'last_login_days_ago',
                'bvn_verified', 'nin_linked', 'telecom_provider']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            data = {}
            for col in feature_cols:
                if col in cat_cols:
                    data[col] = request.form[col]
                elif col in ['has_mobile_app', 'uses_agent', 'loan_active', 'savings_goal_met', 'bvn_verified', 'nin_linked']:
                    data[col] = 1 if request.form.get(col) == 'on' else 0
                else:
                    data[col] = float(request.form[col])

            # Create input DataFrame
            df_input = pd.DataFrame([data])

            # Encode categorical columns
            for col in cat_cols:
                # Transform the single value
                encoded_val = encoder[col].transform([df_input[col].iloc[0]])[0]
                df_input[col] = encoded_val

            # Prepare features for scaling (all columns now numeric)
            X_input = df_input[feature_cols].values  # Order must match training

            # Scale
            X_scaled = scaler.transform(X_input)

            # Predict
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1] * 100  # Churn probability in %

            churn_status = "Likely to Churn" if prediction == 1 else "Likely to Stay"
            risk_level = "High Risk" if probability > 50 else "Low Risk"

            return render_template('index.html',
                                   prediction=churn_status,
                                   probability=f"{probability:.1f}%",
                                   risk_level=risk_level,
                                   show_result=True)

        except Exception as e:
            error_msg = f"Error processing prediction: {str(e)}"
            return render_template('index.html', error=error_msg, show_result=True)

    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    app.run(debug=True)
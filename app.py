from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# === Load your trained model, encoder, and scaler ===
model = joblib.load('financial_churn_model.pkl')
encoder = joblib.load('encoder.pkl')        # Should be a dict: {'state': LabelEncoder, 'account_type': ..., ...}
scaler = joblib.load('scaler.pkl')

# === Feature columns (must match training exactly) ===
cat_cols = ['state', 'account_type', 'monthly_income_bracket', 'preferred_channel']

feature_cols = [
    'age', 'state', 'account_type', 'monthly_income_bracket',
    'avg_monthly_balance', 'num_transactions_month', 'mobile_app_logins',
    'customer_support_calls', 'preferred_channel', 'tenure_months',
    'credit_score'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    show_result = False
    prediction = None
    probability = None
    risk_level = None

    if request.method == 'POST':
        try:
            # Collect form data
            data = {}
            for col in feature_cols:
                value = request.form.get(col)
                if value is None or value == '':
                    raise ValueError(f"Missing value for {col}")
                
                if col in cat_cols:
                    data[col] = value.strip()
                else:
                    data[col] = float(value)

            # Create DataFrame
            df_input = pd.DataFrame([data])

            # === Encode categorical features ===
            for col in cat_cols:
                le = encoder[col]  # Get the correct LabelEncoder for this column
                # Handle unknown categories gracefully
                try:
                    encoded = le.transform([df_input[col].iloc[0]])[0]
                except ValueError:
                    # If unseen category → assign most common class from training (or 0)
                    encoded = 0
                df_input[col] = encoded

            # === Scale numerical features ===
            X = df_input[feature_cols].values.astype(float)
            X_scaled = scaler.transform(X)

            # === Predict ===
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]  # Probability of churn (class 1)

            # Format results
            prediction = "Likely to Churn" if pred == 1 else "Likely to Stay"
            probability = prob  # We'll format in template
            risk_level = "High" if prob > 0.6 else "Medium" if prob > 0.4 else "Low"

            show_result = True

        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            show_result = False

    return render_template(
        'index.html',
        show_result=show_result,
        prediction=prediction,
        probability=probability,        # raw float (e.g. 0.78)
        risk_level=risk_level,
        error=error,
        now=datetime.now()              # for timestamp
    )

if __name__ == '__main__':
    app.run(debug=True)
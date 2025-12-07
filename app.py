from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this'  # Required for sessions

# === Load your trained model, encoder, and scaler ===
model = joblib.load('financial_churn_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# === Feature columns (must match training exactly) ===
cat_cols = ['state', 'account_type', 'monthly_income_bracket', 'preferred_channel']

feature_cols = [
    'age', 'state', 'account_type', 'monthly_income_bracket',
    'avg_monthly_balance', 'num_transactions_month', 'mobile_app_logins',
    'customer_support_calls', 'preferred_channel', 'tenure_months',
    'credit_score'
]

@app.route('/', methods=['GET'])
def index():
    # Retrieve results from session if they exist
    show_result = session.get('show_result', False)
    prediction = session.get('prediction')
    probability = session.get('probability')
    risk_level = session.get('risk_level')
    error = session.get('error')
    
    # Clear session data after displaying
    session.pop('show_result', None)
    session.pop('prediction', None)
    session.pop('probability', None)
    session.pop('risk_level', None)
    session.pop('error', None)

    return render_template(
        'index.html',
        show_result=show_result,
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        error=error,
        now=datetime.now()
    )

@app.route('/predict', methods=['POST'])
def predict():
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
            le = encoder[col]
            try:
                encoded = le.transform([df_input[col].iloc[0]])[0]
            except ValueError:
                encoded = 0
            df_input[col] = encoded

        # === Scale numerical features ===
        X = df_input[feature_cols].values.astype(float)
        X_scaled = scaler.transform(X)

        # === Predict ===
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        # Format results
        prediction = "Likely to Churn" if pred == 1 else "Likely to Stay"
        probability = prob
        risk_level = "High" if prob > 0.6 else "Medium" if prob > 0.4 else "Low"

        # Store results in session
        session['show_result'] = True
        session['prediction'] = prediction
        session['probability'] = probability
        session['risk_level'] = risk_level

    except Exception as e:
        session['error'] = f"Prediction failed: {str(e)}"
        session['show_result'] = False

    # Redirect to GET route (prevents resubmission on refresh)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
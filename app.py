from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import joblib
from datetime import datetime
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this'

model = joblib.load('financial_churn_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

cat_cols = ['state', 'account_type', 'monthly_income_bracket', 'preferred_channel']

feature_cols = [
    'age', 'state', 'account_type', 'monthly_income_bracket',
    'avg_monthly_balance', 'num_transactions_month', 'mobile_app_logins',
    'customer_support_calls', 'preferred_channel', 'tenure_months',
    'credit_score'
]

def process_prediction(df_input):
    df_processed = df_input.copy()
    
    for col in cat_cols:
        le = encoder[col]
        df_processed[col] = df_processed[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
    
    X = df_processed[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return predictions, probabilities

@app.route('/', methods=['GET'])
def index():
    show_result = session.get('show_result', False)
    prediction = session.get('prediction')
    probability = session.get('probability')
    risk_level = session.get('risk_level')
    error = session.get('error')
    
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
# Single prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {}
        for col in feature_cols:
            value = request.form.get(col)
            if value is None or value == '':
                raise ValueError(f"Missing value for {col}")
            
            if col in cat_cols:
                data[col] = value.strip()
            else:
                data[col] = float(value)

        df_input = pd.DataFrame([data])
        predictions, probabilities = process_prediction(df_input)
        
        pred = predictions[0]
        prob = probabilities[0]

        prediction = "Likely to Churn" if pred == 1 else "Likely to Stay"
        probability = prob
        risk_level = "High" if prob > 0.6 else "Medium" if prob > 0.4 else "Low"

        session['show_result'] = True
        session['prediction'] = prediction
        session['probability'] = probability
        session['risk_level'] = risk_level

    except Exception as e:
        session['error'] = f"Prediction failed: {str(e)}"
        session['show_result'] = False

    return redirect(url_for('index'))

# Batch prediction route
@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            session['error'] = "No file uploaded"
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            session['error'] = "No file selected"
            return redirect(url_for('index'))
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            session['error'] = "Invalid file format. Please upload CSV or Excel file"
            return redirect(url_for('index'))
        
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            session['error'] = f"Missing required columns: {', '.join(missing_cols)}"
            return redirect(url_for('index'))
        
        predictions, probabilities = process_prediction(df[feature_cols])
        
        df['churn_prediction'] = predictions
        df['churn_probability'] = probabilities
        df['risk_level'] = df['churn_probability'].apply(
            lambda x: 'High' if x > 0.6 else 'Medium' if x > 0.4 else 'Low'
        )
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'churn_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        
    except Exception as e:
        session['error'] = f"Batch prediction failed: {str(e)}"
        return redirect(url_for('index'))
    
# template download route
@app.route('/download-template')
def download_template():
    sample_data = {
        'age': [35, 42],
        'state': ['Lagos', 'FCT'],
        'account_type': ['Savings', 'Current'],
        'monthly_income_bracket': ['NGN 150,000 - NGN 300,000', 'Greater than NGN 600,000'],
        'avg_monthly_balance': [250000, 1500000],
        'num_transactions_month': [15, 25],
        'mobile_app_logins': [20, 30],
        'customer_support_calls': [2, 1],
        'preferred_channel': ['Mobile App', 'Branch'],
        'tenure_months': [24, 48],
        'credit_score': [650, 750]
    }
    
    df = pd.DataFrame(sample_data)
    
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='churnguard_template.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
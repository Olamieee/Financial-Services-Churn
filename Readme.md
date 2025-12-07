# ChurnGuard AI 🎯

**Predictive Customer Intelligence for Nigerian Banks**

ChurnGuard AI is a machine learning-powered web application that predicts customer churn risk for financial institutions in Nigeria. Built with Flask and scikit-learn, it provides real-time predictions to help banks retain valuable customers.

## 📋 Features

- **Real-time Predictions**: Get churn probability in under 3 seconds
- **Batch Processing**: Upload CSV/Excel files for bulk predictions
- **High Accuracy**: 95.2% prediction accuracy
- **Nigerian Market Focus**: Tailored for Nigerian banking customers across all 36 states
- **User-Friendly Interface**: Clean, responsive design that works on all devices
- **Actionable Insights**: Get retention recommendations based on risk levels
- **Risk Categorization**: Automatic classification into High, Medium, or Low risk
- **Template Download**: Sample file provided for easy batch uploads

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Olamieee/Financial-Services-Churn.git
cd churnguard-ai
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
Navigate to `http://127.0.0.1:5000`

## 📦 Project Structure

```
churnguard-ai/
├── app.py                          # Flask application
├── templates/
│   └── index.html                  # Frontend template
├── financial_churn_model.pkl       # Trained ML model
├── encoder.pkl                     # Label encoders for categorical features
├── scaler.pkl                      # Feature scaler
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

## 🔧 Configuration

### Secret Key

Before deploying to production, change the secret key in `app.py`:

```python
app.secret_key = 'your-secure-random-key-here'
```

Generate a secure key using:
```python
import secrets
print(secrets.token_hex(32))
```

## 📊 Input Features

The model requires the following customer information:

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Customer age (18-100) |
| State | Categorical | Nigerian state of residence (36 states + FCT) |
| Account Type | Categorical | Business, Current, or Savings |
| Monthly Income Bracket | Categorical | 5 income ranges from <₦50k to >₦600k |
| Avg Monthly Balance | Numeric | Average account balance in Naira |
| Transactions per Month | Numeric | Number of monthly transactions |
| Mobile App Logins | Numeric | Monthly app login count |
| Support Calls | Numeric | Customer support calls in last 6 months |
| Preferred Channel | Categorical | ATM, Branch, Mobile App, or USSD |
| Tenure | Numeric | Account age in months |
| Credit Score | Numeric | Customer credit score (300-900) |

## 🎯 Model Performance

- **Accuracy**: 95.2%
- **Response Time**: ≤3 seconds

## 💡 Usage Example

### Single Prediction

1. Fill in customer details in the form
2. Click "Predict Churn Risk"
3. Review the prediction results:
   - **Churn Probability**: Likelihood of customer leaving
   - **Retention Chance**: Probability of customer staying
   - **Risk Level**: High (>60%), Medium (40-60%), or Low (<40%)
4. Follow the retention recommendations

### Batch Prediction

1. Click on "Batch Upload" tab
2. Download the sample template (optional)
3. Prepare your CSV or Excel file with all required columns
4. Upload your file
5. Click "Process Batch Predictions"
6. Download the results file with predictions for all customers

The results file includes:
- All original customer data
- `churn_prediction`: 0 (Stay) or 1 (Churn)
- `churn_probability`: Percentage likelihood (0-100%)
- `risk_level`: High, Medium, or Low

### Risk Level Actions

- **High Risk (>60%)**: Immediate intervention needed
  - Offer personalized bonus rates
  - Provide fee waivers
  - Assign dedicated relationship manager

- **Medium Risk (40-60%)**: Proactive engagement
  - Send targeted offers via SMS
  - Push app notifications
  - Schedule check-in calls

- **Low Risk (<40%)**: Maintain relationship
  - Continue excellent service
  - Standard engagement protocols

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, pandas, joblib
- **Frontend**: HTML5, CSS3, JavaScript
- **File Processing**: openpyxl (Excel), pandas (CSV)
- **Styling**: Custom responsive CSS
- **Session Management**: Flask sessions

## 📝 Requirements

Create a `requirements.txt` file with:

```
gunicorn
flask
pandas
joblib
numpy
scikit-learn
seaborn
matplotlib
openpyxl
```

## 🚢 Deployment

### Local Development
```bash
python app.py
```

### Production (using Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Environment Variables (Production)
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
```

## 🔒 Security Considerations

- Use environment variables for secret keys in production
- Implement rate limiting for API endpoints
- Add input validation and sanitization
- Use HTTPS in production
- Consider adding authentication for sensitive deployments

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername) -->

## 🙏 Acknowledgments

- Nigerian banking sector for domain insights
- scikit-learn community for ML tools
- Flask community for web framework

## 📧 Contact

For questions or support, please reach out:

- **Email**: alongeola16@gmail.com
- **LinkedIn**: [Alonge Olamide](https://www.linkedin.com/in/alonge-olamide-493237242)
---

**Built with ❤️ for Nigerian Financial Institutions**
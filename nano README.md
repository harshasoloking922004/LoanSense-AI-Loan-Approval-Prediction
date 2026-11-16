# LoanSense AI – Loan Approval Prediction

## Overview

LoanSense AI is a machine learning-based web application that predicts loan approval outcomes based on applicant information.

The application integrates a trained Scikit-learn machine learning pipeline with a Streamlit interface to provide interactive loan prediction functionality.

## Features

- User Registration and Login
- Secure Password Hashing
- Loan Approval Prediction
- Machine Learning Model Integration
- Prediction History
- Interactive Analytics Dashboard
- SQLite Database Integration
- Streamlit Web Application

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Streamlit | Web Application |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| Joblib | Model Loading |
| Plotly | Data Visualization |
| SQLite | Database |

## Project Structure

```text
Loan-Approval-Prediction/
│
├── app.py
├── artifacts/
│   └── loan_model.pkl
├── data/
├── reports/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project

```bash
cd Loan-Approval-Prediction
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 4. Activate the Environment

Linux / Ubuntu:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

Open:

http://localhost:8501

## Machine Learning Model

The project uses a trained Scikit-learn Pipeline stored in:

`artifacts/loan_model.pkl`

The model processes applicant information and predicts the loan approval outcome.

## Input Features

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

## Deployment

The application is deployed using Streamlit hosting.

**Live Demo:** [Add your Streamlit application URL]

## Disclaimer

This application is developed for educational and demonstration purposes. Predictions do not guarantee actual loan approval by financial institutions.

## Author

**[Your Name]**

Computer Science and Engineering

## License

[Choose and add an appropriate project license.]

# Student-dropout-rate-prediction
https://student-dropout-rate-prediction-lake.vercel.app/

# Student Retention Analytics Platform

A high-performance machine learning web application powered by **Flask**, **Scikit-Learn**, and **Chart.js**. This platform performs real-time inference using a trained **Logistic Regression** model (`LogisticRegression.pkl`) to predict student dropout risks and visualize key behavioral metrics.

---

## 🎨 Key Features

* **Glassmorphism Analytics UI**: Ultra-clean dark theme built with modern CSS glassmorphism, responsive grids, and micro-interactions.
* **8 Theme Options**: Real-time theme switching (Midnight Glass, Cyberpunk, Deep Forest, Glacial Ice, Magic Purple, Solar Flare, Abyssal Void, and Neon Synthwave).
* **Interactive Visualizations**:
  * **Confidence Donut Chart**: Shows retention vs. dropout probability.
  * **Behavioral Radar Chart**: Plots key student metrics like GPA, Attendance, Study Hours, and Delay.
  * **Model Weights Bar Chart**: Displays feature impacts based on logistic regression coefficients.
* **Multi-Cloud Ready**: Pre-configured for deployment on **Render** (via Gunicorn) or **Vercel** (via `@vercel/python`).

---

## 📁 Repository Structure

```text
├── app.py                  # Main Flask backend & embedded HTML template
├── LogisticRegression.pkl  # Trained Scikit-Learn model binary
├── requirements.txt        # Python dependencies
├── vercel.json             # Vercel deployment configuration
└── README.md               # Project documentation
🛠️ Local Development Setup
1. Prerequisites
Python 3.9 or higher installed.

2. Clone & Setup
Bash
# Clone the repository
git clone [https://github.com/your-username/student-retention-analytics.git](https://github.com/your-username/student-retention-analytics.git)
cd student-retention-analytics

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
3. Run Locally
Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000/.

🚀 Deployment Instructions
Option A: Deploy to Render (Recommended for Flask WSGI)
Push your repository to GitHub or GitLab.

Log into Render and click New + -> Web Service.

Connect your repository.

Configure the build settings:

Runtime: Python

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

Click Create Web Service.

Option B: Deploy to Vercel
Install the Vercel CLI locally (optional) or use the Vercel Web Dashboard.

Push your code to GitHub.

Import the repository into Vercel.

Vercel will automatically detect vercel.json and deploy using the @vercel/python serverless runtime.

📊 Model & API Endpoints
GET /
Renders the analytics dashboard dashboard.

POST /predict
Accepts a JSON payload containing student parameters and returns prediction metrics.

Sample Input Payload:

JSON
{
  "Student_ID": 1001,
  "Age": 20,
  "Gender": 1,
  "Family_Income": 45000,
  "Internet_Access": 1,
  "Study_Hours_per_Day": 3.5,
  "Attendance_Rate": 85.0,
  "Assignment_Delay_Days": 2,
  "Travel_Time_Minutes": 30,
  "Part_Time_Job": 0,
  "Scholarship": 1,
  "Stress_Index": 5.5,
  "GPA": 3.2,
  "Semester_GPA": 3.1,
  "CGPA": 3.15,
  "Semester": 4,
  "Department": 2,
  "Parental_Education": 3
}
Sample Response:

JSON
{
  "prediction": "Student Will Retain",
  "prediction_code": 0,
  "probability_dropout": 12.45,
  "probability_stay": 87.55,
  "risk_level": "Low"
}

✈️ Travel Product Prediction System

A machine learning–powered web application that predicts whether a customer is likely to purchase a travel product based on demographic, behavioral, and sales-pitch attributes.

Built with Python, Scikit-learn, XGBoost, and Streamlit.

📌 Problem Statement

Travel companies spend significant resources pitching travel packages to customers.
However, not all customers are equally likely to convert.

This project aims to:

Predict customer purchase intent

Reduce unnecessary sales efforts

Improve targeting and conversion rate

🧠 Solution Overview

The system uses trained classification models to predict purchase likelihood based on customer information collected during sales interaction.

Key aspects:

Feature preprocessing using a saved pipeline

Probability-based prediction

Interactive web UI for real-time inference

🚀 Features

Interactive Streamlit dashboard

Machine learning–based prediction

Random Forest & XGBoost models

Automatic preprocessing using ColumnTransformer

Probability score for decision support

Modular & scalable project structure


📊 Input Features

The model takes the following inputs:

Age

Type of Contact

City Tier

Duration of Pitch

Occupation

Gender

Number of Followups

Product Pitched

Preferred Property Star

Marital Status

Number of Trips

Passport

Pitch Satisfaction Score

Own Car

Designation

Monthly Income

Total Visiting

📈 Output

Prediction:

0 → Not likely to purchase

1 → Likely to purchase

Probability Score indicating confidence level

⚙️ Installation & Setup
1️⃣ Clone Repository

git clone https://github.com/your-username/travel.git
cd travel

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run Application

streamlit run app.py

🧪 Machine Learning Models Used

Random Forest Classifier

XGBoost Classifier

Feature preprocessing using Scikit-learn Pipelines

Models are pre-trained and loaded using pickle.

🛠 Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

Streamlit

🔮 Future Enhancements

Model comparison toggle (RF vs XGBoost)

SHAP / LIME model explainability

Automated retraining pipeline

Dockerization

Cloud deployment (Streamlit Cloud / AWS)

Logging & monitoring

👨‍💻 Author

Bibek Poudel
Aspiring Data Scientist & ML Engineer
Focused on building production-ready ML systems

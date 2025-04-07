# Credit-Card-fraud-Detection-Model-

This project focuses on detecting fraudulent credit card transactions using various machine learning techniques. With the increasing number of online transactions, it's critical to identify and prevent fraud in real-time. This model is trained on a highly imbalanced dataset, where fraudulent transactions are significantly fewer than legitimate ones.

🔍 Project Overview
The goal of this project is to build a reliable fraud detection model that can:

Accurately classify transactions as fraudulent or non-fraudulent

Handle imbalanced datasets effectively

Minimize false negatives to prevent financial losses

📊 Dataset
Source: Kaggle - Credit Card Fraud Detection Dataset

Description: Contains transactions made by European cardholders in September 2013. The dataset consists of 284,807 transactions, with only 492 (0.172%) being fraudulent.

Features: 30 numerical features (V1-V28 are PCA transformed), Time, Amount, and the target variable Class (1 for fraud, 0 for normal).

⚙️ Workflow
Data Preprocessing

Handled missing/null values

Scaled features using StandardScaler

Addressed class imbalance using SMOTE or undersampling

Exploratory Data Analysis (EDA)

Transaction distribution visualization

Correlation matrix heatmap

Class distribution insights

Model Building
Trained and evaluated multiple classification models:

Logistic Regression

Decision Tree

Random Forest

XGBoost

Support Vector Machine (SVM)

Model Evaluation Metrics

Accuracy, Precision, Recall, F1-Score

Confusion Matrix

ROC Curve & AUC Score

Hyperparameter Tuning
Used GridSearchCV / RandomizedSearchCV for optimal model performance.

🧠 Technologies Used
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Imbalanced-learn (for SMOTE)

XGBoost

📈 Results
Achieved a high recall rate to ensure minimal false negatives

Improved performance on imbalanced data with SMOTE and careful model selection

Final model chosen based on best F1-score and AUC-ROC value

✅ Future Improvements
Integrate real-time detection using a streaming service like Kafka

Deploy the model with Flask/Django API

Add dashboard for visualization using Streamlit or Power BI

📁 Folder Structure
bash
Copy
Edit
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks for EDA and model training
├── models/                # Trained model files (pickle or joblib)
├── visuals/               # Graphs and plots generated during EDA
├── requirements.txt       # List of Python packages
├── fraud_detection.py     # Main script for model training and prediction
└── README.md              # Project description
🤝 Contributions
Feel free to fork this repo and suggest improvements! Pull requests are welcome.

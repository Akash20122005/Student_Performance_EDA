🎓 Student Performance Prediction using EDA & Deep Learning Regression
📘 Overview

This project explores the Student Performance Dataset from the UCI Machine Learning Repository.
The goal is to analyze key academic and personal factors that influence student success and to predict final grades (G3) using a Deep Learning Regression (MLP) model.

Through Exploratory Data Analysis (EDA) and neural network modeling, this study identifies strong predictors of performance such as G1, G2, study time, and absences, helping educational institutions identify at-risk students early.

📂 Project Structure
Student_Performance/
│
├── student+performance.zip             # Original dataset (from UCI)
├── student_performance_unzipped/       # Extracted CSV files
│   └── student-mat.csv                 # Main dataset used
│
├── student_mlp_regressor.h5            # Trained MLP model
├── preprocessing_pipeline.joblib       # Saved preprocessing pipeline
│
├── student_performance.ipynb           # Main notebook/code
├── README.md                           # Project documentation
└── requirements.txt                    # Dependencies

🧠 Key Steps in the Project
1️⃣ Data Loading & Cleaning

Dataset loaded from UCI archive.

Checked for missing values and duplicates.

Handled outliers using IQR method.

2️⃣ Exploratory Data Analysis (EDA)

Histogram: Distribution of final grades (G3)

Correlation Heatmap: Relationship among numeric features

Boxplots: G3 comparison by gender

Scatter plots: G1 vs G3

Bar charts: Student counts by school

3️⃣ Data Preprocessing

Numeric features: Median imputation + Standard scaling

Categorical features: Mode imputation + One-hot encoding

Combined using ColumnTransformer and Pipeline

4️⃣ Model Building (MLP)

Implemented Multi-Layer Perceptron (MLP) with:

ReLU activation

Dropout regularization

Adam optimizer (variable learning rate)

Hyperparameters tuned manually for best validation RMSE.

5️⃣ Evaluation

Metrics used:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

Visualization:

Loss & MAE curves

Predicted vs Actual

Residual plots

📊 Results Summary
Metric	Value
Test RMSE	~1.9
Test MAE	~1.4
Test R²	~0.89

✅ The MLP model achieved high predictive accuracy, confirming its suitability for regression tasks in educational analytics.

🚀 How to Run the Project
🔧 Requirements

Create a requirements.txt file:

pandas
numpy
matplotlib
scikit-learn
tensorflow
joblib

▶️ Steps

Clone or download the project.

Place student+performance.zip in the project directory.

Run the Python script or Jupyter notebook:

python student_performance.py


or open in Colab / Jupyter:

jupyter notebook student_performance.ipynb


The model and preprocessing pipeline will be saved as:

student_mlp_regressor.h5

preprocessing_pipeline.joblib

🧩 Conclusion

This project successfully demonstrates the integration of EDA and Deep Learning Regression for educational data analytics.
Strong predictors such as G1, G2, study time, and absences significantly impact final performance.
The MLP model effectively captures nonlinear relationships and provides robust grade predictions.

🔮 Future Scope

Incorporate behavioral and psychological factors (motivation, stress).

Test ensemble models (XGBoost, Gradient Boosting) for comparison.

Build an interactive dashboard for performance tracking.

Extend analysis to multi-school or regional datasets.

Integrate Explainable AI (SHAP/LIME) for interpretability.

📚 References

UCI Machine Learning Repository – Student Performance Dataset

Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.

Chollet, F. (2018). Deep Learning with Python. Manning Publications.

Raschka, S., & Mirjalili, V. (2019). Python Machine Learning. Packt Publishing.

McKinney, W. (2017). Python for Data Analysis. O’Reilly Media.

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.

Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90–95.

# DataPulse: Analytics & ML Dashboard

DataPulse is an interactive Machine Learning and Data Analytics dashboard built using Streamlit. It allows users to upload datasets, perform automated data cleaning, explore data through interactive visualizations, train various machine learning models (both regression and classification), evaluate their performance, and make predictions—all from a user-friendly web interface without writing a single line of code.

## Features

### 1. Data Ingestion & Analysis
- **File Upload:** Upload any CSV dataset.
- **Automated Data Cleaning:**
  - Removes null/missing values.
  - Removes duplicate rows.
  - Removes outliers using the Interquartile Range (IQR) method for numeric columns.
- **Data Preview & Statistics:** View the raw data shape, data samples, and descriptive statistics.
- **Exploratory Data Analytics (EDA):** Create interactive visualizations using Seaborn or Plotly.
  - **Seaborn:** Correlation Heatmaps, Box Plots, Scatter Plots.
  - **Plotly:** Bar Charts, Scatter Plots, Pie Charts, Line Charts, Histograms.

### 2. Model Training & Evaluation
- **Preprocessing Pipeline:**
  - Automatic One-Hot Encoding for categorical variables.
  - Optional Standard Scaling (`StandardScaler`) for features.
- **Supported Models:**
  - **Regression:** Linear Regression, Polynomial Regression, K-Nearest Neighbors (KNN), Decision Tree, Support Vector Machine (SVM).
  - **Classification:** Logistic Regression, Polynomial (Logistic), K-Nearest Neighbors (KNN), Decision Tree, Support Vector Machine (SVM).
- **Model Evaluation:**
  - 5-Fold Cross Validation.
  - **Regression Metrics:** R-Squared (R²), Mean Absolute Error (MAE), Mean Squared Error (MSE), and Predicted vs Actual plots.
  - **Classification Metrics:** Accuracy, Precision, Recall, F1-Score, Classification Report, Confusion Matrix, and ROC Curve (for binary classification).
  - **Feature Importance:** Visualized for Decision Tree models.
- **Model Deployment:** Download the trained model as a `.pkl` file for external use.
- **Model Comparison Dashboard:** Instantly train and compare the performance of multiple models on your dataset.

### 3. Prediction
- **Interactive Inference:** Enter feature values directly into the UI.
- **Real-time Prediction:** Utilize the model trained in memory to predict target values on-the-fly.

## Installation

1. **Clone the repository or download the source code.**
2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies include: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`.*

## Usage

To start the Streamlit application, run the following command in your terminal from the project directory:

```bash
python -m streamlit run app.py
```
*(Alternatively, you can run the command specified in `run.txt`: `d:\Model_Selection\venv\Scripts\python.exe -m streamlit run app.py`)*

This will start a local server, and your default web browser will automatically open the DataPulse Dashboard.

## Architecture & Workarounds
- The application implements a specialized meta-path finder workaround in `app.py` to bypass Application Control policy blocks related to `pyarrow` DLLs, ensuring smooth execution in restricted environments.

## License
[MIT License](LICENSE) (or choose an appropriate license for your project).

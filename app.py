import sys
import types

# --- Workaround: pyarrow DLL blocked by Application Control policy ---
# scikit-learn 1.8+ and pandas 3.x deeply import pyarrow, whose native
# DLL is blocked on this system.  We install a meta-path finder that
# intercepts *every* pyarrow.* import and returns a lightweight mock
# module where any attribute access returns a safe dummy object.

class _MockPyarrowModule(types.ModuleType):
    """Module whose every attribute is a no-op callable / returns itself."""
    def __getattr__(self, name):
        # Return a callable dummy that also supports attribute access
        return _MockPyarrowModule(f"{self.__name__}.{name}")
    def __call__(self, *a, **kw):
        return None

class _PyarrowFinder:
    """sys.meta_path finder that captures any 'pyarrow' or 'pyarrow.*' import."""
    def find_module(self, fullname, path=None):
        if fullname == 'pyarrow' or fullname.startswith('pyarrow.'):
            return self
        return None
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _MockPyarrowModule(fullname)
        mod.__path__ = []                # treat every sub-module as a package
        if fullname == 'pyarrow':
            mod.__version__ = '17.0.0'
        sys.modules[fullname] = mod
        return mod

sys.meta_path.insert(0, _PyarrowFinder())

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Evaluation Metrics
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_curve, auc
)

# Set page config
st.set_page_config(page_title="DataPulse: Analytics & ML", layout="wide")

st.title("DataPulse: Analytics & ML Dashboard")
st.markdown("Upload your dataset, explore it with interactive analytics, and train Machine Learning models with ease!")

# --- Session State Initialization ---
if 'clean_df' not in st.session_state:
    st.session_state.clean_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'cat_cols' not in st.session_state:
    st.session_state.cat_cols = None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Go to", ["Data Analysis", "Model Training", "Prediction"])
st.sidebar.markdown("---")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# ==========================================
# 1. DATA ANALYSIS TAB
# ==========================================
if nav == "Data Analysis":
    st.header("1. Data Ingestion & Analysis")
    
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    
    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)
        df = raw_df.copy()
        
        st.subheader("Data Cleaning (Auto-Removal)")
        initial_shape = df.shape
        
        # 1. Null values
        null_count = int(df.isnull().sum().sum())
        if null_count > 0:
            df = df.dropna()
            
        # 2. Duplicate values
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            df = df.drop_duplicates()
            
        # 3. Outliers removal using IQR for numeric columns
        outlier_count = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            outlier_count = int((~outlier_condition).sum())
            if outlier_count > 0:
                df = df[outlier_condition]
                
        st.session_state.clean_df = df
        
        st.markdown("##### Detected Data Quality Issues")
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Null Values", value=null_count, delta="Removed" if null_count > 0 else "Clean", delta_color="inverse" if null_count > 0 else "normal")
        m2.metric(label="Duplicate Rows", value=dup_count, delta="Removed" if dup_count > 0 else "Clean", delta_color="inverse" if dup_count > 0 else "normal")
        m3.metric(label="Outlier Rows", value=outlier_count, delta="Removed" if outlier_count > 0 else "Clean", delta_color="inverse" if outlier_count > 0 else "normal")
        
        st.markdown("---")
        if null_count > 0 or dup_count > 0 or outlier_count > 0:
            st.success(f"✨ **Dataset Cleaned!** Initial Shape: `{initial_shape}` ➔ Final Shape: `{df.shape}`")
        else:
            st.success(f"✨ **Dataset is perfectly clean!** Shape: `{df.shape}`")
        
        st.subheader("Data Preview")
        st.markdown(f'<div style="overflow-x: auto;">{df.head().to_html(index=True)}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
        
        st.subheader("Data Statistics")
        st.markdown(f'<div style="overflow-x: auto;">{df.describe().to_html(index=True)}</div>', unsafe_allow_html=True)
            
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        
        st.header("Exploratory Data Analytics (EDA)")
        st.sidebar.header("Data Analytics Options")
        chart_library = st.sidebar.radio("Chart Library", ["Seaborn", "Plotly"], key="chart_lib")

        numeric_df = df.select_dtypes(include=[np.number])
        all_cols = df.columns.tolist()
        num_cols = numeric_df.columns.tolist()

        if chart_library == "Seaborn":
            seaborn_chart = st.sidebar.selectbox("Choose Seaborn Chart", ["None", "Heatmap", "Box Plot", "Scatter Plot"])
            if seaborn_chart == "Heatmap":
                if len(num_cols) > 1:
                    st.subheader("Correlation Heatmap (Seaborn)")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                    st.pyplot(fig)
            elif seaborn_chart == "Box Plot":
                col_box = st.sidebar.selectbox("Select Column", options=num_cols)
                if col_box:
                    st.subheader(f"Box Plot of {col_box} (Seaborn)")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=df, y=col_box, color='#AB63FA', ax=ax)
                    st.pyplot(fig)
            elif seaborn_chart == "Scatter Plot":
                col_x = st.sidebar.selectbox("X-Axis", options=num_cols, index=0)
                col_y = st.sidebar.selectbox("Y-Axis", options=num_cols, index=min(1, len(num_cols)-1))
                if col_x and col_y:
                    st.subheader(f"Scatter Plot: {col_y} vs {col_x} (Seaborn)")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(data=df, x=col_x, y=col_y, color='#EF553B', ax=ax)
                    st.pyplot(fig)
                    
        else:
            plotly_chart = st.sidebar.selectbox("Choose Plotly Chart", ["None", "Bar Chart", "Scatter Plot", "Pie Chart", "Line Chart", "Histogram"])
            if plotly_chart == "Bar Chart":
                col_x = st.sidebar.selectbox("X-Axis (Category)", options=all_cols, index=0)
                col_y = st.sidebar.selectbox("Y-Axis (Value)", options=num_cols, index=0)
                if col_x and col_y:
                    st.subheader(f"Bar Chart: {col_y} by {col_x} (Plotly)")
                    fig = px.bar(df, x=col_x, y=col_y, color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig, use_container_width=True)
            elif plotly_chart == "Scatter Plot":
                col_x = st.sidebar.selectbox("X-Axis", options=all_cols, index=0)
                col_y = st.sidebar.selectbox("Y-Axis", options=all_cols, index=min(1, len(all_cols)-1))
                if col_x and col_y:
                    st.subheader(f"Scatter Plot: {col_y} vs {col_x} (Plotly)")
                    fig = px.scatter(df, x=col_x, y=col_y, color_discrete_sequence=['#EF553B'])
                    st.plotly_chart(fig, use_container_width=True)
            elif plotly_chart == "Pie Chart":
                col_pie = st.sidebar.selectbox("Category Column", options=all_cols)
                if col_pie:
                    st.subheader(f"Pie Chart of {col_pie} (Plotly)")
                    pie_data = df[col_pie].value_counts().reset_index()
                    pie_data.columns = [col_pie, 'count']
                    fig = px.pie(pie_data, names=col_pie, values='count')
                    st.plotly_chart(fig, use_container_width=True)
            elif plotly_chart == "Line Chart":
                col_x = st.sidebar.selectbox("X-Axis", options=all_cols, index=0)
                col_y = st.sidebar.selectbox("Y-Axis", options=num_cols, index=0)
                if col_x and col_y:
                    st.subheader(f"Line Chart: {col_y} over {col_x} (Plotly)")
                    fig = px.line(df.sort_values(col_x), x=col_x, y=col_y, color_discrete_sequence=['#00CC96'])
                    st.plotly_chart(fig, use_container_width=True)
            elif plotly_chart == "Histogram":
                col_hist = st.sidebar.selectbox("Select Column", options=all_cols)
                if col_hist:
                    st.subheader(f"Histogram of {col_hist} (Plotly)")
                    fig = px.histogram(df, x=col_hist, marginal="box", color_discrete_sequence=['#FFA15A'])
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Awaiting file upload...")

# ==========================================
# 2. MODEL TRAINING TAB
# ==========================================
elif nav == "Model Training":
    st.header("2. Model Training & Evaluation")
    
    if st.session_state.clean_df is None:
        st.warning("Please upload and clean a dataset in the 'Data Analysis' tab first.")
    else:
        df = st.session_state.clean_df
        columns = df.columns.tolist()
        
        st.subheader("Target & Feature Selection")
        col3, col4 = st.columns(2)
        with col3:
            target_col = st.selectbox("Select Target Variable (Y)", options=columns)
        with col4:
            default_features = [col for col in columns if col != target_col]
            feature_cols = st.multiselect("Select Feature Variables (X)", options=columns, default=default_features)
            
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            st.stop()
            
        X = df[feature_cols]
        y = df[target_col]
        
        # Save to session state
        st.session_state.features = feature_cols
        st.session_state.target = target_col
        
        st.subheader("Preprocessing Pipeline")
        
        cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        st.session_state.cat_cols = cat_cols
        if len(cat_cols) > 0:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            st.success(f"One-Hot Encoded categorical columns: {cat_cols}")

        unique_y = y.nunique()
        is_numeric_y = pd.api.types.is_numeric_dtype(y)
        auto_task = "Classification" if (not is_numeric_y) or (unique_y < 20) else "Regression"
        
        col_task, col_scale = st.columns(2)
        with col_task:
            task_type = st.radio("Task Type", ["Regression", "Classification"], index=0 if auto_task=="Regression" else 1)
            st.session_state.task_type = task_type
            
        with col_scale:
            apply_scaling = st.toggle("Apply StandardScaler", value=False)
            
        if apply_scaling:
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns)
                st.session_state.scaler = scaler
                st.success("StandardScaler applied to features.")
            except Exception as e:
                st.error(f"Error scaling data: {e}. Ensure features are numeric.")
                st.stop()
        else:
            st.session_state.scaler = None

        # Sidebar config
        st.sidebar.header("Model Configuration")
        
        if task_type == "Regression":
            model_options = ["Linear Regression", "Polynomial Regression", "KNN", "Decision Tree", "SVM"]
            model_choice = st.sidebar.selectbox("Choose Model", model_options)
            
            if model_choice == "Polynomial Regression":
                degree = st.sidebar.slider("Degree", 2, 5, 2)
                model = LinearRegression()
            elif model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "KNN":
                n_neighbors = st.sidebar.slider("n_neighbors", 1, 30, 5)
                model = KNeighborsRegressor(n_neighbors=n_neighbors)
            elif model_choice == "Decision Tree":
                max_depth = st.sidebar.slider("max_depth", 1, 30, 5)
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            elif model_choice == "SVM":
                kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
                c_val = st.sidebar.number_input("C parameter", min_value=0.01, max_value=100.0, value=1.0)
                model = SVR(kernel=kernel, C=c_val)
                
        else:
            model_options = ["Logistic Regression", "Polynomial (Logistic)", "KNN", "Decision Tree", "SVM"]
            model_choice = st.sidebar.selectbox("Choose Model", model_options)
            
            if model_choice == "Polynomial (Logistic)":
                degree = st.sidebar.slider("Degree", 2, 5, 2)
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_choice == "KNN":
                n_neighbors = st.sidebar.slider("n_neighbors", 1, 30, 5)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
            elif model_choice == "Decision Tree":
                max_depth = st.sidebar.slider("max_depth", 1, 30, 5)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            elif model_choice == "SVM":
                kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
                c_val = st.sidebar.number_input("C parameter", min_value=0.01, max_value=100.0, value=1.0)
                # Enable probability for ROC curve
                model = SVC(kernel=kernel, C=c_val, probability=True, random_state=42)

        st.sidebar.markdown("---")
        col_run, col_comp = st.columns(2)
        run_model = col_run.button("Train & Evaluate Model", use_container_width=True)
        compare_models = col_comp.button("Compare All Models", use_container_width=True)

        if run_model:
            st.header("Evaluation Results")
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            except Exception as e:
                st.error(f"Error during split: {e}")
                st.stop()
                
            with st.spinner(f"Training {model_choice}..."):
                try:
                    if model_choice in ["Polynomial Regression", "Polynomial (Logistic)"]:
                        poly = PolynomialFeatures(degree=degree)
                        X_train_poly = poly.fit_transform(X_train)
                        X_test_poly = poly.transform(X_test)
                        model.fit(X_train_poly, y_train)
                        y_pred = model.predict(X_test_poly)
                        cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                except ValueError as e:
                    st.error(f"Error during training: {e}. This usually happens if you chose a Regression model for a column containing text/categories. Try selecting 'Classification' instead!")
                    st.stop()
                except Exception as e:
                    st.error(f"An unexpected error occurred during training: {e}")
                    st.stop()
                
                st.session_state.model = model # Save to session state
                
            st.success(f"Model trained successfully! 5-Fold Cross Validation Mean Score: {cv_scores.mean():.4f}")

            # --- Regression Evaluation ---
            if task_type == "Regression":
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                met1, met2, met3 = st.columns(3)
                met1.metric(label="R-Squared (R²)", value=f"{r2:.4f}")
                met2.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                met3.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                
                st.subheader("Predicted vs Actual")
                fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'})
                fig.add_shape(type="line", line=dict(dash='dash'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                st.plotly_chart(fig, use_container_width=True)
                
            # --- Classification Evaluation ---
            else:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc:.4f}")
                m2.metric("Precision", f"{prec:.4f}")
                m3.metric("Recall", f"{rec:.4f}")
                m4.metric("F1-Score", f"{f1:.4f}")
                
                col_rep, col_cm = st.columns(2)
                with col_rep:
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.markdown(pd.DataFrame(report).transpose().to_html(index=True), unsafe_allow_html=True)
                    
                with col_cm:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                
                # ROC Curve for binary classification
                if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
                    st.subheader("ROC Curve")
                    try:
                        y_prob = model.predict_proba(X_test if "Polynomial" not in model_choice else X_test_poly)[:, 1]
                        fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=np.unique(y)[1])
                        roc_auc = auc(fpr, tpr)
                        
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))
                        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                        st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception as e:
                        st.info("ROC Curve not available for this model/data configuration.")

            # Feature Importance
            if model_choice == "Decision Tree" and hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance = model.feature_importances_
                # Match features names properly if scaling/polynomial were applied
                # For simplicity, we assume standard features or scaled features (names preserved)
                try:
                    feat_names = X.columns
                    feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
                    fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Decision Tree Feature Importance")
                    fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_feat, use_container_width=True)
                except Exception as e:
                    st.warning("Could not plot feature importances (feature name mismatch).")

            # Download Model
            st.markdown("---")
            st.subheader("Deployment")
            model_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            st.download_button(
                label="📥 Download Trained Model (.pkl)",
                data=model_buffer.getvalue(),
                file_name=f"{model_choice.replace(' ', '_').lower()}.pkl",
                mime="application/octet-stream"
            )

        elif compare_models:
            st.header("Model Comparison Dashboard")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            results = []

            if task_type == "Regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "KNN": KNeighborsRegressor(n_neighbors=5),
                    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
                    "SVM": SVR(kernel='linear', C=1.0)
                }
                
                try:
                    for name, m in models.items():
                        m.fit(X_train, y_train)
                        preds = m.predict(X_test)
                        results.append({"Model": name, "Score (R²)": r2_score(y_test, preds)})
                except ValueError as e:
                    st.error(f"Error during regression comparison: {e}. If your target contains text, switch to Classification.")
                    st.stop()
            else:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
                    "SVM": SVC(kernel='linear', C=1.0, random_state=42)
                }
                try:
                    for name, m in models.items():
                        m.fit(X_train, y_train)
                        preds = m.predict(X_test)
                        results.append({"Model": name, "Score (Accuracy)": accuracy_score(y_test, preds)})
                except Exception as e:
                    st.error(f"Error during classification comparison: {e}")
                    st.stop()
                    
            res_df = pd.DataFrame(results)
            res_df = res_df.sort_values(by=res_df.columns[1], ascending=False)
            st.markdown(res_df.to_html(index=False), unsafe_allow_html=True)
            
            fig_comp = px.bar(res_df, x="Model", y=res_df.columns[1], color="Model", title="Model Performance Comparison")
            st.plotly_chart(fig_comp, use_container_width=True)

# ==========================================
# 3. PREDICTION TAB
# ==========================================
elif nav == "Prediction":
    st.header("3. Make Predictions")
    
    if st.session_state.model is None:
        st.warning("Please train a model in the 'Model Training' tab first!")
    else:
        st.success("Model loaded from memory. Enter feature values to predict.")
        
        input_data = {}
        for col in st.session_state.features:
            if pd.api.types.is_numeric_dtype(st.session_state.clean_df[col]):
                mean_val = float(st.session_state.clean_df[col].mean())
                input_data[col] = st.number_input(f"{col}", value=mean_val)
            else:
                options = st.session_state.clean_df[col].unique()
                input_data[col] = st.selectbox(f"{col}", options)
                
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            
            # Apply same preprocessing
            if st.session_state.cat_cols:
                input_df = pd.get_dummies(input_df)
                missing_cols = set(st.session_state.model.feature_names_in_) - set(input_df.columns)
                for c in missing_cols:
                    input_df[c] = 0
                input_df = input_df[st.session_state.model.feature_names_in_]
                
            if st.session_state.scaler:
                input_scaled = st.session_state.scaler.transform(input_df)
                input_df = pd.DataFrame(input_scaled, columns=input_df.columns)
                
            prediction = st.session_state.model.predict(input_df)[0]
            st.markdown(f"### Predicted {st.session_state.target}: `{prediction}`")

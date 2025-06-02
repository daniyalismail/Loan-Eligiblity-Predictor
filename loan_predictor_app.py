import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(5,3.5)) # Slightly adjusted for sidebar
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_ticklabels(classes, fontsize=8)
    ax.yaxis.set_ticklabels(classes, fontsize=8)
    plt.tight_layout() # Adjust layout to prevent labels from being cut off
    return fig

# Function to load and preprocess data
@st.cache_data # Cache data loading and preprocessing
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as the script or provide the correct path.")
        return None, None, None, None, None, None, None
    except pd.errors.EmptyDataError:
        st.error(f"Error: The file '{file_path}' is empty. Please provide a valid CSV file.")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        return None, None, None, None, None, None, None

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Expected columns check
    expected_cols = ['loan_id', 'no_of_dependents', 'education', 'self_employed', 
                     'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value', 'loan_status']
    missing_expected_cols = [col for col in expected_cols if col not in df.columns and col != 'loan_id'] # loan_id is optional for dropping
    if 'loan_status' not in df.columns:
        st.error("Critical Error: The target column 'loan_status' is missing from the dataset. Cannot proceed.")
        return None, None, None, None, None, None, None
    if missing_expected_cols:
        st.warning(f"Warning: The following expected feature columns are missing: {', '.join(missing_expected_cols)}. This might affect model training or input fields.")


    # Drop 'loan_id' if it exists
    if 'loan_id' in df.columns:
        df = df.drop('loan_id', axis=1)
    
    # Clean string values in categorical columns (remove leading/trailing spaces)
    categorical_cols_to_clean = ['education', 'self_employed', 'loan_status']
    for col in categorical_cols_to_clean:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.strip()
        elif col in df.columns and df[col].dtype != 'object':
             st.warning(f"Warning: Column '{col}' was expected to be of type 'object' (string) for mapping but found {df[col].dtype}. Mapping might fail or be incorrect.")
        elif col not in df.columns and col == 'loan_status': # Already handled above, but good to be cautious
            pass # Error already raised
        elif col not in df.columns:
            st.warning(f"Warning: Categorical column '{col}' not found for cleaning/mapping.")


    # --- Preprocessing ---
    # Handle categorical features
    try:
        if 'education' in df.columns:
            df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})
        if 'self_employed' in df.columns:
            df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
        if 'loan_status' in df.columns: # Should always be true due to earlier check
            df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    except Exception as e:
        st.error(f"Error during categorical mapping: {e}. Please check if the values in 'education', 'self_employed', or 'loan_status' columns are as expected (e.g., 'Graduate', 'Not Graduate', 'Yes', 'No', 'Approved', 'Rejected').")
        return None, None, None, None, None, None, None

    # Check for NaN values created by mapping (if unexpected values were present)
    cols_to_check_nan = [col for col in ['education', 'self_employed', 'loan_status'] if col in df.columns]
    if df[cols_to_check_nan].isnull().any().any():
        st.warning("Warning: Missing values detected after mapping categorical features. This might be due to unexpected values in these columns. Rows with NaN in these critical columns will be dropped for model training.")
        df.dropna(subset=cols_to_check_nan, inplace=True)
        if df.empty:
            st.error("Error: Dataset became empty after dropping rows with mapping issues. Please check your CSV file's categorical column values.")
            return None, None, None, None, None, None, None
    
    # Handle any other NaNs in feature columns by imputation (e.g., mean for numerical)
    # For simplicity, we'll fill with mean for numerical, mode for others if any remain.
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
                st.info(f"Filled NaN in numerical column '{col}' with mean.")
            else: # Should not happen for mapped columns if handled above
                df[col].fillna(df[col].mode()[0], inplace=True)
                st.info(f"Filled NaN in categorical column '{col}' with mode.")


    # Define features (X) and target (y)
    if 'loan_status' not in df.columns: # Should be caught earlier
        st.error("Target 'loan_status' is missing.")
        return None, None, None, None, None, None, None
        
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    feature_names = X.columns.tolist() # Get feature names before scaling for consistent input fields

    # Identify numerical columns for scaling
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError as ve:
        if "The least populated class in y has only 1 member, which is too few." in str(ve):
            st.error(f"ValueError during data split: {ve}. This often means one of the classes in 'loan_status' has too few samples after preprocessing. Consider a larger dataset or different preprocessing for very small classes.")
            return None, None, None, None, None, None, None
        else:
            st.error(f"Error during data split: {ve}")
            return None, None, None, None, None, None, None


    # Scale numerical features
    scaler = StandardScaler()
    if numerical_cols: # Only scale if there are numerical columns
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    else:
        st.warning("No numerical columns found to scale.")
    

    return X_train, X_test, y_train, y_test, scaler, feature_names, numerical_cols

# --- Main App ---
st.set_page_config(layout="wide", page_title="Loan Predictor", page_icon="💰")

st.title("🏦 Loan Eligibility Predictor")
st.markdown("---")

with st.expander("ℹ️ Instructions to Use", expanded=False):
    st.markdown("""
    1.  This application predicts loan eligibility based on the provided applicant details.
    2.  The prediction model is trained using a dataset (`loan_approval_dataset.csv`).
    3.  Performance metrics of the model (Accuracy, Confusion Matrix, Classification Report) are available in the sidebar.
    4.  Fill in all the required fields in the '📝 Enter Applicant Details' section below.
    5.  Click the "Evaluate Eligibility" button.
    6.  The result will indicate **Approved (1)** or **Not Approved (0)**.
    """)

# Load data and train model
file_path = 'loan_approval_dataset.csv' 
X_train, X_test, y_train, y_test, scaler, feature_names, numerical_cols = load_and_preprocess_data(file_path)

model = None
accuracy = 0
conf_matrix = None
class_report_str = None # Renamed for clarity (string version)

if X_train is not None and not X_train.empty and y_train is not None and not y_train.empty:
    try:
        model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced') # Added class_weight
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report_str = classification_report(y_test, y_pred, target_names=['Not Approved (0)', 'Approved (1)'])

    except Exception as e:
        st.error(f"An error occurred during model training or evaluation: {e}")
        model = None 
else:
    if X_train is None or X_train.empty:
        st.error("Model training could not proceed: Training data is not available or is empty after preprocessing.")

# --- Sidebar ---
st.sidebar.header("📊 Model Performance")
if model is not None and accuracy > 0:
    st.sidebar.subheader("Accuracy Score")
    st.sidebar.metric(label="Accuracy", value=f"{accuracy*100:.2f}%", delta=None) # Using st.metric for a nicer look

    st.sidebar.subheader("Confusion Matrix")
    if conf_matrix is not None:
        fig_cm_sidebar = plot_confusion_matrix(conf_matrix, classes=['No (0)', 'Yes (1)'], title="Confusion Matrix")
        st.sidebar.pyplot(fig_cm_sidebar)

    st.sidebar.subheader("Classification Report")
    if class_report_str is not None:
        st.sidebar.code(class_report_str)
else:
    st.sidebar.warning("Model not trained or metrics not available.")

# --- Main Interface for User Input ---
st.header("📝 Enter Applicant Details for Prediction")

if model is None:
    st.warning("⚠️ Model is not available for prediction. Please check data loading and training steps or potential errors reported above.")
else:
    input_data = {}
    if feature_names: # Ensure feature_names is populated
        # Create two columns for a cleaner layout of input fields
        col1, col2 = st.columns(2)
        
        # Distribute input fields across columns
        for i, feature in enumerate(feature_names):
            current_container = col1 if i < (len(feature_names) + 1) // 2 else col2
            with current_container:
                try:
                    # Use more descriptive labels and provide guidance where possible
                    label = feature.replace("_", " ").title()
                    if feature == 'education':
                        input_data[feature] = st.selectbox(f'{label}', options=[0, 1], format_func=lambda x: 'Not Graduate' if x == 0 else 'Graduate', key=f"input_{feature}", help="Applicant's education level.")
                    elif feature == 'self_employed':
                        input_data[feature] = st.selectbox(f'{label}', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key=f"input_{feature}", help="Is the applicant self-employed?")
                    elif feature == 'no_of_dependents':
                        input_data[feature] = st.number_input(f'{label}', min_value=0, max_value=20, value=0, step=1, key=f"input_{feature}", help="Number of dependents.")
                    elif feature == 'loan_term': 
                        input_data[feature] = st.number_input(f'{label} (Years)', min_value=1, max_value=40, value=10, step=1, key=f"input_{feature}", help="Loan term in years.")
                    elif feature == 'cibil_score':
                        input_data[feature] = st.number_input(f'{label}', min_value=300, max_value=900, value=700, step=10, key=f"input_{feature}", help="Applicant's CIBIL score (credit score).")
                    elif 'income_annum' == feature:
                         input_data[feature] = st.number_input(f'{label} (Pkr)', min_value=0.0, value=500000.0, step=10000.0, format="%.0f", key=f"input_{feature}", help="Annual income in Rupees.")
                    elif 'loan_amount' == feature:
                         input_data[feature] = st.number_input(f'{label} (Pkr)', min_value=0.0, value=1000000.0, step=50000.0, format="%.0f", key=f"input_{feature}", help="Requested loan amount in Rupees.")
                    elif 'value' in feature : # For other asset values
                         input_data[feature] = st.number_input(f'{label} (Pkr)', min_value=0.0, value=100000.0, step=10000.0, format="%.0f", key=f"input_{feature}", help=f"Value of {label.lower()} in Rupees.")
                    else: # Default number input for any other numerical features (should be rare if dataset is as expected)
                        input_data[feature] = st.number_input(f'{label}', value=0.0, format="%.2f", key=f"input_{feature}")
                except Exception as e:
                    st.error(f"Error creating input field for {feature}: {e}")
                    input_data[feature] = 0 # Default value in case of error
    else:
        st.error("Feature names are not available from the dataset. Cannot create input fields for prediction.")

    st.markdown("---") # Separator before the button
    
    # Prediction button - centered
    button_col1, button_col2, button_col3 = st.columns([2,1,2]) # Create 3 columns, button in middle one
    with button_col2:
        predict_button = st.button("🚀 Evaluate Eligibility", key="predict_button", use_container_width=True)

    if predict_button:
        if not feature_names:
            st.error("Cannot predict as feature names are not available.")
        elif not input_data:
             st.error("Input data is empty. Please fill the form.")
        else:
            try:
                input_df = pd.DataFrame([input_data])[feature_names] # Ensure column order

                if scaler and numerical_cols:
                    input_df_scaled = input_df.copy()
                    # Only transform if numerical_cols is not empty and columns exist in input_df_scaled
                    cols_to_scale = [col for col in numerical_cols if col in input_df_scaled.columns]
                    if cols_to_scale:
                        input_df_scaled[cols_to_scale] = scaler.transform(input_df_scaled[cols_to_scale])
                    elif numerical_cols: # numerical_cols has items, but none are in input_df_scaled
                        st.warning("Numerical columns identified for scaling, but none found in the current input data structure. Prediction might be inaccurate.")
                else:
                    st.warning("Scaler or numerical column list not available/applicable. Using raw inputs for prediction (might be inaccurate if model was trained on scaled data).")
                    input_df_scaled = input_df.copy() # Use unscaled if scaler isn't ready


                if input_df_scaled is not None:
                    prediction = model.predict(input_df_scaled)

                    st.markdown("---")
                    st.subheader("🎯 Prediction Result:")
                    if prediction[0] == 1:
                        st.success("🎉 Loan Approved (Status: 1)")
                        st.balloons()
                    else:
                        st.error("😞 Loan Not Approved (Status: 0)")
                    # st.markdown("---")


            except ValueError as ve:
                if "Input contains NaN" in str(ve):
                    st.error(f"Prediction Error: {ve}. One or more input fields might be missing or invalid after processing. Please check all inputs.")
                elif "features were seen during fit" in str(ve):
                     st.error(f"Prediction Error: {ve}. This usually means there's a mismatch between the features the model was trained on and the features provided for prediction. Check if all input fields are correctly mapped to the dataset columns.")
                else:
                    st.error(f"An error occurred during prediction: {ve}")
                st.error("Please ensure all inputs are valid numbers where required and all fields are filled.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                st.error("Please check your inputs and try again.")

st.markdown("<div style='text-align: center; color: grey;'>Developed By M.Daniyal Ismail (Roll No:39) & Syed Maazin Imran (Roll No:23)</div>", unsafe_allow_html=True)

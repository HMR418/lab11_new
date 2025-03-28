import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------------------------------
# 1. Define a function to replicate the preprocessing 
#    done in training
# -------------------------------------------------------
def preprocess_data(input_df):
    """
    This function replicates (simplifies) the steps you took during training.
    In your actual code, you must ensure the same transformations:
     1) Fill missing values for numeric and categorical columns
     2) Identify numeric and categorical features
     3) Scale or encode appropriately
    """
    df = input_df.copy()

    # Replace 'Max_Load_Tons' with your actual target column name
    # if you are uploading a dataset that still has the target column.
    target_col = 'Max_Load_Tons'
    if target_col in df.columns:
        df.drop(columns=[target_col], inplace=True)

    # Fill missing
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Separate categorical and numeric features
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(exclude=['object']).columns.tolist()

    # Build pipelines
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Fit_transform the data — in production, you’d typically *fit* the pipeline 
    # once on your training set, then *transform* new data.
    # But for simplicity here, we just do it directly.
    X_processed = preprocessor.fit_transform(df)

    return X_processed

# -------------------------------------------------------
# 2. Main Streamlit Application
# -------------------------------------------------------
def main():
    st.title("Bridge Load Prediction App")

    st.write("""
    **Instructions**:
    1. Upload a file with the same columns used in training (including `Max_Load_Tons` if you want to drop it before inference).
    2. This app will run the same preprocessing logic and then use the **tf_bridge_model.h5** file to predict.
    """)

    uploaded_file = st.file_uploader("Upload the bridge data Excel file", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        # Read in the file
        if uploaded_file.name.endswith(".xlsx"):
            input_df = pd.read_excel(uploaded_file)
        else:
            input_df = pd.read_csv(uploaded_file)
        
        st.subheader("Raw Input Data")
        st.write(input_df.head())

        # Preprocess the data
        X_processed = preprocess_data(input_df)

        # Load the trained model
        try:
            model = tf.keras.models.load_model("tf_bridge_model.h5")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Make predictions
        predictions = model.predict(X_processed)

        st.subheader("Predictions")
        # Depending on your model output, you may want to round or format
        st.write(predictions)
    else:
        st.warning("Please upload a file first.")

if __name__ == "__main__":
    main()

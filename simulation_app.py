import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, List, Tuple

# --- Configuration ---
MODEL_PATH = 'ids_model.pkl'

# --- Helper Functions (Adapted from your simulate_input.py) ---

@st.cache_resource
def load_model_data(path=MODEL_PATH):
    """
    Loads the model file which contains the model, selected features,
    scaler, and encoders.
    """
    if not os.path.exists(path):
        st.error(f"Error: Model file not found at {path}")
        st.stop()
    
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Validate model components
        required_keys = ['model', 'selected', 'scaler', 'encoders']
        if not all(key in data for key in required_keys):
            st.error("Error: Model file is missing required components (model, selected, scaler, encoders).")
            st.stop()
            
        return data
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.stop()

def build_row_from_inputs(selected: List[str], encoders: Dict, scaler, user_vals: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a single-row DataFrame from user inputs, applying encoding,
    filling defaults, and preparing it for scaling.
    
    Adapted from simulate_input.py
    """
    row = {}
    
    # Get scaler mean for numerical defaults if available
    scaler_mean = getattr(scaler, 'mean_', np.zeros(len(selected)))
    
    for i, col in enumerate(selected):
        user_val = user_vals.get(col)

        if user_val is None or str(user_val).strip() == '':
            # User left this blank, use default
            if col in encoders:
                # Default to first class in encoder
                row[col] = 0  # index 0
            else:
                # Default to scaler mean for numerical
                row[col] = scaler_mean[i]
        else:
            # User provided a value
            if col in encoders:
                # Find the encoded value
                le = encoders[col]
                try:
                    # Find index of the user's string value
                    val_index = list(le.classes_).index(user_val)
                    row[col] = val_index
                except ValueError:
                    # Value not in encoder (e.g., new category), default to 0
                    row[col] = 0 
            else:
                # Numerical feature
                try:
                    row[col] = float(user_val)
                except ValueError:
                    # Invalid number, default to scaler mean
                    row[col] = scaler_mean[i]
                    
    df_row = pd.DataFrame([row], columns=selected)
    return df_row

def predict_simulation(model, scaler, df_row: pd.DataFrame, selected: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales the input row and returns model predictions and probabilities.
    
    Adapted from simulate_input.py
    """
    try:
        # Ensure columns are in the correct order
        df_row_ordered = df_row[selected]
        
        # Scale the data
        scaled_data = scaler.transform(df_row_ordered)
        
        # Get predictions
        preds = model.predict(scaled_data)
        
        # Get probabilities
        probs = model.predict_proba(scaled_data)
        
        return preds, probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# --- Streamlit App UI ---

st.set_page_config(page_title="IDS Simulator", layout="wide")
st.title("🛡️ Network Intrusion Detection Simulator")
st.markdown("Use this app to simulate a network connection's features and predict if it's an attack or normal traffic using the pre-trained model.")

# Load model data
model_data = load_model_data(MODEL_PATH)
model = model_data['model']
selected_features = model_data['selected']
scaler = model_data['scaler']
encoders = model_data['encoders']
class_encoder = encoders.get('class')

if not class_encoder:
    st.error("Error: 'class' encoder not found in model file. Cannot determine output labels.")
    st.stop()

st.header("Simulate Connection Features")
st.markdown(f"Please provide values for the {len(selected_features)} features required by the model. You can leave fields blank to use a default value.")

# Create a form for inputs
with st.form(key='simulation_form'):
    user_inputs = {}
    
    # Use columns to make the form more compact
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, feature in enumerate(selected_features):
        col = cols[i % num_cols]
        
        if feature in encoders:
            # Categorical feature: Use a dropdown (selectbox)
            # Add a blank option for default
            options = [''] + list(encoders[feature].classes_)
            user_inputs[feature] = col.selectbox(f"{feature} (categorical)", options=options, key=feature)
        else:
            # Numerical feature: Use a text input
            user_inputs[feature] = col.text_input(f"{feature} (numerical)", key=feature, placeholder="e.g., 0.0 (leave blank for default)")

    submit_button = st.form_submit_button(label='Detect Intrusion')

# Handle form submission
if submit_button:
    # 1. Collect inputs from the form (already in user_inputs dict)
    # Convert empty strings to None for default handling
    processed_inputs = {f: (user_inputs[f] if user_inputs[f] != '' else None) for f in selected_features}
    
    # 2. Build the single-row DataFrame
    st.write("Processing inputs...")
    df_row = build_row_from_inputs(selected_features, encoders, scaler, processed_inputs)

    # 3. Get prediction
    preds, probs = predict_simulation(model, scaler, df_row, selected_features)
    
    if preds is not None and probs is not None:
        prediction_index = preds[0]
        probabilities = probs[0]
        
        # Get human-readable label
        try:
            prediction_label = class_encoder.classes_[prediction_index]
        except IndexError:
            st.error("Error: Model predicted an unknown class index.")
            st.stop()

        st.header("Detection Result")
        
        # Display the main result
        if prediction_label.lower() == 'anomaly':
            st.error(f"🚨 Attack Detected! (Prediction: {prediction_label})", icon="🔥")
        else:
            st.success(f"✅ Connection is Normal (Prediction: {prediction_label})", icon="🛡️")
        
        # Display probabilities
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame(probabilities, index=class_encoder.classes_, columns=['Probability'])
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
        st.dataframe(prob_df)
        
        with st.expander("See Raw Input Data (as sent to model)"):
            st.write("This is the raw, encoded, and default-filled data row before scaling:")
            st.dataframe(df_row)

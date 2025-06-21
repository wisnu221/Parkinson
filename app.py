import numpy as np
import pandas as pd
import streamlit as st
import pickle

# --- Load the trained model ---
# Pastikan 'parkinsons_model.sav' berada di direktori yang sama dengan 'app.py'
try:
    loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Error: 'parkinsons_model.sav' not found. Please ensure the model file is in the same directory.")
    st.stop() # Stop the app if the model isn't found

# --- List of features (columns) expected by the model ---
# Ini penting untuk memastikan urutan input sesuai dengan urutan fitur saat model dilatih.
# Anda bisa mendapatkan ini dari X.columns.tolist() setelah dropping 'name' and 'status'
# Dari output notebook Anda, ini adalah daftar kolomnya:
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

# --- Streamlit App Interface ---
st.set_page_config(page_title="Parkinson's Disease Prediction", page_icon="🧠")
st.title("Parkinson's Disease Prediction App")
st.markdown("Enter the patient's voice parameters to predict if they have Parkinson's disease.")

# Input fields for the user
# Anda bisa membuat input sesuai dengan 22 fitur yang ada
col1, col2, col3 = st.columns(3)

with col1:
    fo = st.text_input('MDVP:Fo(Hz) (Average vocal fundamental frequency)')
    fhi = st.text_input('MDVP:Fhi(Hz) (Maximum vocal fundamental frequency)')
    flo = st.text_input('MDVP:Flo(Hz) (Minimum vocal fundamental frequency)')
    jitter_percent = st.text_input('MDVP:Jitter(%) (Measure of variation in fundamental frequency)')
    jitter_abs = st.text_input('MDVP:Jitter(Abs) (Absolute measure of variation in fundamental frequency)')
    rap = st.text_input('MDVP:RAP (Relative Average Perturbation)')
    ppq = st.text_input('MDVP:PPQ (Five-point Period Perturbation Quotient)')
    ddp = st.text_input('Jitter:DDP (DDP of Jitter)') # This is Jitter:DDP as per your notebook output

with col2:
    shimmer = st.text_input('MDVP:Shimmer (Measure of amplitude variation)')
    shimmer_db = st.text_input('MDVP:Shimmer(dB) (Shimmer in dB)')
    shimmer_apq3 = st.text_input('Shimmer:APQ3 (Three-point Amplitude Perturbation Quotient)')
    shimmer_apq5 = st.text_input('Shimmer:APQ5 (Five-point Amplitude Perturbation Quotient)')
    mdvp_apq = st.text_input('MDVP:APQ (Measure of amplitude variation with respect to average amplitude)')
    shimmer_dda = st.text_input('Shimmer:DDA (DDA of Shimmer)')
    nhr = st.text_input('NHR (Noise-to-Harmonic Ratio)')
    hnr = st.text_input('HNR (Harmonic-to-Noise Ratio)')

with col3:
    rpde = st.text_input('RPDE (Recurrence Period Density Entropy)')
    dfa = st.text_input('DFA (Detrended Fluctuation Analysis)')
    spread1 = st.text_input('spread1 (Nonlinear measure of fundamental frequency variation)')
    spread2 = st.text_input('spread2 (Nonlinear measure of fundamental frequency variation)')
    d2 = st.text_input('D2 (Correlation Dimension)')
    ppe = st.text_input('PPE (Pitch Period Entropy)')

# Button for prediction
parkinsons_diagnosis = ''

# Check if all inputs are provided before attempting conversion
input_provided = all([fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                      shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq,
                      shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe])

if st.button('Parkinson\'s Test Result'):
    if not input_provided:
        st.warning("Please fill in all the input fields.")
    else:
        try:
            # Convert input strings to floats
            input_data = [
                float(fo), float(fhi), float(flo), float(jitter_percent),
                float(jitter_abs), float(rap), float(ppq), float(ddp),
                float(shimmer), float(shimmer_db), float(shimmer_apq3),
                float(shimmer_apq5), float(mdvp_apq), float(shimmer_dda),
                float(nhr), float(hnr), float(rpde), float(dfa),
                float(spread1), float(spread2), float(d2), float(ppe)
            ]

            # Reshape the numpy array for prediction
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            # Make prediction
            prediction = loaded_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                parkinsons_diagnosis = "The Person does not have Parkinson's Disease"
            else:
                parkinsons_diagnosis = "The Person has Parkinson's Disease"
            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("Invalid input. Please ensure all inputs are numerical values.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Optional: Add information about the features
st.sidebar.header("About the Features")
st.sidebar.info(
    """
    These features are various voice measurements that can indicate the presence of Parkinson's disease.
    For example:
    - **MDVP:Fo(Hz)**: Average vocal fundamental frequency.
    - **MDVP:Jitter(%)**: Measure of variation in fundamental frequency.
    - **MDVP:Shimmer**: Measure of amplitude variation.
    - **HNR**: Harmonic-to-Noise Ratio.
    """
)

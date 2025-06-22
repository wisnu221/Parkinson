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

# --- List of features with their corresponding descriptions and default values ---
# Menggunakan dictionary untuk menyimpan nama fitur, deskripsi, dan nilai default
# Nilai default ini bisa disesuaikan dengan mean atau median dari dataset Anda
# atau nilai yang masuk akal secara medis.
features_info = {
    "MDVP:Fo(Hz)": {"label": "1. MDVP:Fo(Hz) (Rata-rata frekuensi fundamental vokal)", "default": 154.23},
    "MDVP:Fhi(Hz)": {"label": "2. MDVP:Fhi(Hz) (Frekuensi fundamental vokal maksimum)", "default": 197.10},
    "MDVP:Flo(Hz)": {"label": "3. MDVP:Flo(Hz) (Frekuensi fundamental vokal minimum)", "default": 116.32},
    "MDVP:Jitter(%)": {"label": "4. MDVP:Jitter(%) (Ukuran variasi frekuensi fundamental)", "default": 0.0062},
    "MDVP:Jitter(Abs)": {"label": "5. MDVP:Jitter(Abs) (Ukuran absolut variasi frekuensi fundamental)", "default": 0.000044},
    "MDVP:RAP": {"label": "6. MDVP:RAP (Relative Average Perturbation)", "default": 0.0033},
    "MDVP:PPQ": {"label": "7. MDVP:PPQ (Five-point Period Perturbation Quotient)", "default": 0.0034},
    "Jitter:DDP": {"label": "8. Jitter:DDP (DDP dari Jitter)", "default": 0.0099},
    "MDVP:Shimmer": {"label": "9. MDVP:Shimmer (Ukuran variasi amplitudo)", "default": 0.0297},
    "MDVP:Shimmer(dB)": {"label": "10. MDVP:Shimmer(dB) (Shimmer dalam dB)", "default": 0.2823},
    "Shimmer:APQ3": {"label": "11. Shimmer:APQ3 (Three-point Amplitude Perturbation Quotient)", "default": 0.0165},
    "Shimmer:APQ5": {"label": "12. Shimmer:APQ5 (Five-point Amplitude Perturbation Quotient)", "default": 0.0179},
    "MDVP:APQ": {"label": "13. MDVP:APQ (Ukuran variasi amplitudo terhadap amplitudo rata-rata)", "default": 0.0241},
    "Shimmer:DDA": {"label": "14. Shimmer:DDA (DDA dari Shimmer)", "default": 0.0470},
    "NHR": {"label": "15. NHR (Noise-to-Harmonic Ratio)", "default": 0.0248},
    "HNR": {"label": "16. HNR (Harmonic-to-Noise Ratio)", "default": 21.89},
    "RPDE": {"label": "17. RPDE (Recurrence Period Density Entropy)", "default": 0.4985},
    "DFA": {"label": "18. DFA (Detrended Fluctuation Analysis)", "default": 0.7181},
    "spread1": {"label": "19. spread1 (Ukuran nonlinier variasi frekuensi fundamental)", "default": -5.6844},
    "spread2": {"label": "20. spread2 (Ukuran nonlinier variasi frekuensi fundamental)", "default": 0.2265},
    "D2": {"label": "21. D2 (Dimensi Korelasi)", "default": 2.3818},
    "PPE": {"label": "22. PPE (Pitch Period Entropy)", "default": 0.2066},
}

# Urutan fitur sesuai saat melatih model
# Ini sangat penting! Pastikan urutan ini cocok dengan X.columns saat model dilatih.
feature_order = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

# --- Streamlit App Interface ---
st.set_page_config(page_title="Parkinson's Disease Prediction", page_icon="🧠", layout="wide")
st.title("Aplikasi Prediksi Penyakit Parkinson")
st.markdown("Masukkan parameter suara pasien untuk memprediksi apakah mereka mengidap penyakit Parkinson.")

# Dictionary to store user inputs
user_inputs = {}

# Create input fields in expanders for better organization
with st.expander("Parameter Frekuensi Vokal (MDVP:Fo, Fhi, Flo, Jitter, RAP, PPQ, DDP)"):
    col1, col2 = st.columns(2)
    with col1:
        user_inputs["MDVP:Fo(Hz)"] = st.number_input(features_info["MDVP:Fo(Hz)"]["label"], value=features_info["MDVP:Fo(Hz)"]["default"], format="%.3f")
        user_inputs["MDVP:Fhi(Hz)"] = st.number_input(features_info["MDVP:Fhi(Hz)"]["label"], value=features_info["MDVP:Fhi(Hz)"]["default"], format="%.3f")
        user_inputs["MDVP:Flo(Hz)"] = st.number_input(features_info["MDVP:Flo(Hz)"]["label"], value=features_info["MDVP:Flo(Hz)"]["default"], format="%.3f")
        user_inputs["MDVP:Jitter(%)"] = st.number_input(features_info["MDVP:Jitter(%)"]["label"], value=features_info["MDVP:Jitter(%)"]["default"], format="%.5f")
    with col2:
        user_inputs["MDVP:Jitter(Abs)"] = st.number_input(features_info["MDVP:Jitter(Abs)"]["label"], value=features_info["MDVP:Jitter(Abs)"]["default"], format="%.6f")
        user_inputs["MDVP:RAP"] = st.number_input(features_info["MDVP:RAP"]["label"], value=features_info["MDVP:RAP"]["default"], format="%.5f")
        user_inputs["MDVP:PPQ"] = st.number_input(features_info["MDVP:PPQ"]["label"], value=features_info["MDVP:PPQ"]["default"], format="%.5f")
        user_inputs["Jitter:DDP"] = st.number_input(features_info["Jitter:DDP"]["label"], value=features_info["Jitter:DDP"]["default"], format="%.5f")

with st.expander("Parameter Amplitudo Vokal (MDVP:Shimmer, Shimmer(dB), APQ, DDA)"):
    col1, col2 = st.columns(2)
    with col1:
        user_inputs["MDVP:Shimmer"] = st.number_input(features_info["MDVP:Shimmer"]["label"], value=features_info["MDVP:Shimmer"]["default"], format="%.5f")
        user_inputs["MDVP:Shimmer(dB)"] = st.number_input(features_info["MDVP:Shimmer(dB)"]["label"], value=features_info["MDVP:Shimmer(dB)"]["default"], format="%.3f")
        user_inputs["Shimmer:APQ3"] = st.number_input(features_info["Shimmer:APQ3"]["label"], value=features_info["Shimmer:APQ3"]["default"], format="%.5f")
    with col2:
        user_inputs["Shimmer:APQ5"] = st.number_input(features_info["Shimmer:APQ5"]["label"], value=features_info["Shimmer:APQ5"]["default"], format="%.5f")
        user_inputs["MDVP:APQ"] = st.number_input(features_info["MDVP:APQ"]["label"], value=features_info["MDVP:APQ"]["default"], format="%.5f")
        user_inputs["Shimmer:DDA"] = st.number_input(features_info["Shimmer:DDA"]["label"], value=features_info["Shimmer:DDA"]["default"], format="%.5f")

with st.expander("Parameter Suara Lainnya (NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE)"):
    col1, col2 = st.columns(2)
    with col1:
        user_inputs["NHR"] = st.number_input(features_info["NHR"]["label"], value=features_info["NHR"]["default"], format="%.5f")
        user_inputs["HNR"] = st.number_input(features_info["HNR"]["label"], value=features_info["HNR"]["default"], format="%.3f")
        user_inputs["RPDE"] = st.number_input(features_info["RPDE"]["label"], value=features_info["RPDE"]["default"], format="%.6f")
        user_inputs["DFA"] = st.number_input(features_info["DFA"]["label"], value=features_info["DFA"]["default"], format="%.6f")
    with col2:
        user_inputs["spread1"] = st.number_input(features_info["spread1"]["label"], value=features_info["spread1"]["default"], format="%.6f")
        user_inputs["spread2"] = st.number_input(features_info["spread2"]["label"], value=features_info["spread2"]["default"], format="%.6f")
        user_inputs["D2"] = st.number_input(features_info["D2"]["label"], value=features_info["D2"]["default"], format="%.6f")
        user_inputs["PPE"] = st.number_input(features_info["PPE"]["label"], value=features_info["PPE"]["default"], format="%.6f")

# Button for prediction
parkinsons_diagnosis = ''

if st.button("Dapatkan Hasil Tes Parkinson"):
    try:
        # Create a list of input values in the correct order
        input_data = [user_inputs[feature_name] for feature_name in feature_order]

        # Reshape the numpy array for prediction
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            parkinsons_diagnosis = "Orang tersebut **TIDAK** mengidap Penyakit Parkinson."
        else:
            parkinsons_diagnosis = "Orang tersebut **MENGIDAP** Penyakit Parkinson."
        st.success(parkinsons_diagnosis)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input atau membuat prediksi: {e}")
        st.info("Pastikan semua nilai yang dimasukkan sudah benar.")

# Optional: Add information about the features
st.sidebar.header("Tentang Fitur-Fitur Suara")
st.sidebar.info(
    """
    Fitur-fitur ini adalah berbagai pengukuran suara yang dapat mengindikasikan adanya penyakit Parkinson.
    Setiap fitur memberikan informasi tentang karakteristik vokal yang berbeda:
    - **MDVP:Fo(Hz)**: Rata-rata frekuensi fundamental vokal.
    - **MDVP:Jitter(%)**: Ukuran variasi dalam frekuensi fundamental (ketidakstabilan nada).
    - **MDVP:Shimmer**: Ukuran variasi amplitudo (ketidakstabilan volume).
    - **HNR**: Rasio Harmonik-ke-Derau, mengukur tingkat "kebersihan" suara.
    """
)

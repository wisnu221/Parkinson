import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Memuat model yang telah disimpan
try:
    loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Pastikan file 'parkinsons_model.sav' ada di direktori yang sama.")
    st.stop()


# Nama kolom yang digunakan untuk pelatihan model
# Anda bisa mendapatkan ini dari notebook Anda dengan menjalankan `for column in X.columns: print(column)`
feature_names = [
    'fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP', 
    'Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'MDVP_RAP', 'MDVP_PPQ', 
    'Jitter_DDP', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]


# Fungsi untuk prediksi
def parkinsons_prediction(input_data):
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return "The Person does not have Parkinsons Disease"
    else:
        return "The Person has Parkinsons"


# Antarmuka pengguna Streamlit
st.title('Parkinsons Disease Prediction App')

st.write("Masukkan nilai berikut untuk memprediksi:")

# Input fields for each feature
input_values = {}
for feature in feature_names:
    input_values[feature] = st.text_input(f'Enter {feature}:')

# Tombol prediksi
if st.button('Predict'):
    # Validate input
    try:
        input_data = [float(input_values[feature]) for feature in feature_names]
        diagnosis = parkinsons_prediction(input_data)
        st.success(diagnosis)
    except ValueError:
        st.error("Harap masukkan nilai numerik yang valid untuk semua bidang.")

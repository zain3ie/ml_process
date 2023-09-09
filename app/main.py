import requests
import streamlit as st


with st.form('form'):
    st.subheader('Artifitial Moon Object Survival Prediction')
    st.write('''
        Aplikasi ini akan membantu memprediksi apakah peluncuran artifisial objek ke bulan akan berhasil atau tidak.
        Prediksi dilakukan berdasarkan negara pengirim, tahun peluncuran, dan berat dari objek artifisial.
    ''')

    country = st.text_input(label='Negara pengirim')
    year = st.number_input(label='Tahun peluncuran', min_value=1950, max_value=2100)
    mass = st.number_input(label='Berat objek (kg)', min_value=0)

    submit = st.form_submit_button(label='Predict')

    if submit:
        with st.spinner('Predicting...'):
            response = requests.post(f'http://127.0.0.1:8000/predict?country={country}&year={year}&mass={mass}')

        if response.status_code == 422:
            st.warning(response.json()['detail'], icon='❗')
        elif response.status_code == 200:
            if response.json() == 0:
                st.error('Artifisial objek yang dikirim akan hancur', icon='💥')
            elif response.json() == 1:
                st.success('Artifisial objek yang dikirim akan berhasil menjalankan misi', icon='🎉')

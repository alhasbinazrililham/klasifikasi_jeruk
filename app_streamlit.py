import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Belajar Klasifikasi Jeruk", page_icon=":tangerine:")


model = joblib.load("model_klasifikasi_jeruk.joblib")

st.title(":tangerine: Belajar Klasifikasi Jeruk")
st.markdown("Application Machine Learning untuk memprediksi kualitas jeruk")

diameter = st.slider("Diameter", 4.0, 10.0, 6.5)
berat = st.slider("Berat", 70.0, 280.0, 150.0)
tebal_kulit = st.slider("Tebal Kulit", 0.2, 2.0, 0.7)
kadar_gula = st.slider("Kadar Gula", 5.0, 15.0, 10.0)
asal_daerah = st.pills(
    "Asal Daerah", ["Jawa Tengah", "Kalimantan", "Jawa Barat"], default="Kalimantan"
)
warna = st.pills("Warna", ["hijau", "kuning", "oranye"], default="kuning")
musim_panen = st.pills("Musim Panen", ["kemarau", "hujan"], default="hujan")

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame(
        [
            [
                diameter,
                berat,
                tebal_kulit,
                kadar_gula,
                asal_daerah,
                warna,
                musim_panen,
            ]
        ],
        columns=[
            "diameter",
            "berat",
            "tebal_kulit",
            "kadar_gula",
            "asal_daerah",
            "warna",
            "musim_panen",
        ],
    )

    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    st.success(
        f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**"
    )

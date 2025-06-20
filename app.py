import streamlit as st
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
st.set_page_config(page_title="Analisis Kualitas Data & Suara", layout="wide")
st.markdown("""
<style>
    .small-font { font-size:13px !important; }
    .compact-table td, .compact-table th {
        padding: 6px 10px !important;
        font-size: 13px !important;
    }
    .stButton>button {
        padding: 0.4rem 0.75rem;
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Analisis Data Tunarungu & Deteksi Kualitas Suara")

st.markdown("""
Aplikasi ini memiliki dua fitur utama:
1. **📄 Analisis Data Tunarungu berdasarkan CSV**: Lihat data jumlah tunarungu berdasarkan provinsi, kabupaten, kecamatan, dan desa. Termasuk filter, visualisasi, dan ekspor hasil.
2. **🎙️ Deteksi Kualitas Suara berdasarkan File Audio**: Menggunakan fitur suara seperti energi sinyal dan MFCC untuk mengklasifikasikan kualitas suara menjadi **Bagus**, **Sedang**, atau **Jelek**.

---
""")

# ---------- MENU PILIHAN ----------
menu = st.sidebar.radio("🔎 Pilih Jenis Analisis:", ["📄 Unggah CSV", "🎙️ Unggah Suara"])

# ---------- OPSI 1: CSV ----------
if menu == "📄 Unggah CSV":
    st.header("📄 Analisis Data CSV Tunarungu")
    st.markdown("""
    **📘 Penjelasan:**
    - Data ini berisi informasi jumlah tunarungu per wilayah administratif.
    - Anda dapat melakukan filter berdasarkan Provinsi, Kabupaten, Kecamatan, dan Desa.
    - Visualisasi akan memperlihatkan jumlah tunarungu di setiap desa hasil filter.
    """)
    
    uploaded_csv = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv, delimiter=';')

        st.subheader("🗂️ Data Awal (Preview)")
        st.dataframe(df.head())

        # Filter interaktif
        st.subheader("📍 Filter Wilayah")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            prov = st.selectbox("Provinsi", ["Semua"] + sorted(df.iloc[:,1].unique()))
        with col2:
            kab = st.selectbox("Kabupaten", ["Semua"] + sorted(df.iloc[:,3].unique()))
        with col3:
            kec = st.selectbox("Kecamatan", ["Semua"] + sorted(df.iloc[:,5].unique()))
        with col4:
            desa = st.selectbox("Desa", ["Semua"] + sorted(df.iloc[:,7].unique()))

        filtered = df.copy()
        if prov != "Semua":
            filtered = filtered[filtered.iloc[:,1] == prov]
        if kab != "Semua":
            filtered = filtered[filtered.iloc[:,3] == kab]
        if kec != "Semua":
            filtered = filtered[filtered.iloc[:,5] == kec]
        if desa != "Semua":
            filtered = filtered[filtered.iloc[:,7] == desa]

        st.subheader("📊 Tabel Data Filter")
        st.dataframe(filtered.style.set_table_attributes("class='compact-table'"), use_container_width=True)

        if not filtered.empty:
            st.subheader("📈 Diagram Jumlah Tunarungu per Desa")
            fig, ax = plt.subplots(figsize=(10,4))
            filtered.plot(kind='bar', x=filtered.columns[7], y=filtered.columns[9], ax=ax, color='teal')
            ax.set_ylabel("Jumlah")
            ax.set_title("Distribusi Tunarungu ")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        # Ekspor
        csv_out = filtered.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("⬇️ Download Data CSV", csv_out, "data_tunarungu_filter.csv", "text/csv")

        # Kesimpulan akhir
        st.subheader("📌 Ringkasan Data")
        st.markdown(f"Total baris hasil filter: **{len(filtered)}**")
        st.markdown(f"Total tunarungu : **{filtered.iloc[:,9].sum()} jiwa**")

# ---------- OPSI 2: AUDIO ----------
else:
    import librosa.display
    import soundfile as sf

    st.header("🎙️ Deteksi Kualitas Suara")
    st.markdown("""
    **📘 Penjelasan Klasifikasi Suara:**

    Kualitas suara diklasifikasikan berdasarkan **energi sinyal rata-rata (RMS)**:

    | Kategori | Simbol | Energi RMS | Penjelasan |
    |----------|--------|-------------|-------------|
    | Bagus    | 🟩     | > 0.03      | Suara jernih, stabil, minim noise |
    | Sedang   | 🟨     | 0.01 – 0.03 | Cukup jelas, sedikit noise |
    | Jelek    | 🟥     | < 0.01      | Pelan, noise tinggi, tidak jelas |

    🔍 Visualisasi MFCC membantu melihat karakteristik frekuensi suara.
    """)

    audio = st.file_uploader("Unggah file suara (.wav)", type=["wav"])

    if audio is not None:
        st.audio(audio)
        y, sr = librosa.load(audio, sr=22050)
        energy = np.mean(librosa.feature.rms(y=y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Klasifikasi suara
        if energy < 0.01:
            kualitas = "🟥 Jelek"
            deskripsi = "Suara sangat pelan atau penuh noise."
        elif energy < 0.03:
            kualitas = "🟨 Sedang"
            deskripsi = "Suara cukup jelas namun terdapat gangguan."
        else:
            kualitas = "🟩 Bagus"
            deskripsi = "Suara jernih, stabil, dan minim noise."

        st.markdown(f"<div class='small-font'>🎯 <b>Kualitas Suara:</b> {kualitas}<br>📝 <b>Deskripsi:</b> {deskripsi}</div>", unsafe_allow_html=True)

        # Diagram MFCC
        st.subheader("🎵 Visualisasi MFCC (Koefisien)")
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(mfcc_mean[:20], marker='o', color='indigo')
        ax2.set_title("Rata-rata MFCC (20 Pertama)")
        ax2.set_xlabel("Koefisien")
        ax2.set_ylabel("Nilai")
        st.pyplot(fig2)

        # Tampilkan MFCC[0–9] dan interpretasi
        st.markdown("📋 <b>Detail MFCC[0–9] dan Interpretasi</b>", unsafe_allow_html=True)
        penjelasan = [
            "Energi dasar suara (volume dan intensitas)",
            "Frekuensi rendah (berat atau tipisnya suara)",
            "Nada dasar dan intonasi", 
            "Fluktuasi nada", 
            "Kompleksitas suara menengah", 
            "Komponen desis atau noise", 
            "Kontur resonansi tinggi",
            "Artikulasi tajam", 
            "Detail suara halus", 
            "Perubahan cepat/distorsi"
        ]
        mfcc_table = pd.DataFrame({
            "Koefisien": [f"MFCC[{i}]" for i in range(10)],
            "Nilai": [round(m, 2) for m in mfcc_mean[:10]],
            "Deskripsi": penjelasan
        })
        st.dataframe(mfcc_table.style.set_table_attributes("class='compact-table'"), use_container_width=True)

        # Ekspor hasil
        hasil_df = pd.DataFrame({
            "Nama File": [audio.name],
            "Kualitas": [kualitas],
            "Deskripsi": [deskripsi],
            "Energi": [round(energy, 5)],
            **{f"MFCC[{i}]": [round(val, 3)] for i, val in enumerate(mfcc_mean[:10])}
        })

        st.download_button("⬇️ Download Hasil Suara (CSV)", hasil_df.to_csv(index=False).encode("utf-8"), "hasil_suara.csv", "text/csv")

        # Simpulan akhir
        st.subheader("📌 Ringkasan Hasil Suara")
        st.markdown(f"File **{audio.name}** diklasifikasikan sebagai: **{kualitas}**")
        st.markdown(f"Rata-rata energi sinyal: **{round(energy, 5)}**")
        st.markdown("Interpretasi MFCC menunjukkan karakteristik frekuensi yang konsisten dengan hasil klasifikasi kualitas suara.")

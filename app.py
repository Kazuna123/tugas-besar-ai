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
1. **📄 Analisis Data Tunarungu berdasarkan CSV**  
2. **🎙️ Deteksi Kualitas Suara berdasarkan File Audio**

---
""")

# ---------- MENU PILIHAN ----------
menu = st.sidebar.radio("🔎 Pilih Jenis Analisis:", ["📄 Unggah CSV", "🎙️ Unggah Suara"])

# ---------- OPSI 1: CSV ----------
if menu == "📄 Unggah CSV":
    st.header("📄 Analisis Data CSV Tunarungu")
    st.markdown("""
    **📘 Penjelasan:**  
    - Data ini berisi informasi jumlah tunarungu per wilayah.  
    - Bisa difilter berdasarkan Provinsi, Kabupaten, Kecamatan, dan Desa.  
    - Tampilkan grafik dan ekspor CSV.
    """)

    uploaded_csv = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv, delimiter=';')
        st.subheader("🗂️ Data Awal")
        st.dataframe(df.head())

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
            ax.set_title("Distribusi Tunarungu")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        # Ekspor
        csv_out = filtered.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("⬇️ Download Data CSV", csv_out, "data_tunarungu_filter.csv", "text/csv")

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

    | Kategori | Simbol | Energi RMS | Penjelasan |
    |----------|--------|-------------|-------------|
    | Bagus    | 🟩     | > 0.03      | Suara jernih, stabil, minim noise |
    | Sedang   | 🟨     | 0.01 – 0.03 | Cukup jelas, sedikit noise |
    | Jelek    | 🟥     | < 0.01      | Pelan, noise tinggi, tidak jelas |
    """)

    audio = st.file_uploader("Unggah file suara (.wav)", type=["wav"])

    if audio is not None:
        st.audio(audio)
        y, sr = librosa.load(audio, sr=22050)
        energy = np.mean(librosa.feature.rms(y=y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Klasifikasi
        if energy < 0.01:
            kualitas = "🟥 Jelek"
            deskripsi = "Suara sangat pelan atau penuh noise."
        elif energy < 0.03:
            kualitas = "🟨 Sedang"
            deskripsi = "Suara cukup jelas namun ada gangguan."
        else:
            kualitas = "🟩 Bagus"
            deskripsi = "Suara jernih dan stabil."

        st.markdown(f"<div class='small-font'>🎯 <b>Kualitas Suara:</b> {kualitas}<br>📝 <b>Deskripsi:</b> {deskripsi}</div>", unsafe_allow_html=True)

        # Visualisasi MFCC
        st.subheader("🎵 Visualisasi MFCC (Mel-Frequency Cepstral Coefficients)")
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(mfcc_mean[:20], marker='o', color='indigo')
        ax2.set_title("Rata-rata MFCC (20 Pertama)")
        ax2.set_xlabel("Koefisien")
        ax2.set_ylabel("Nilai")
        st.pyplot(fig2)

        # Penjelasan MFCC
        st.markdown("📋 <b>Apa Itu MFCC?</b>", unsafe_allow_html=True)
        st.markdown("""
        **MFCC (Mel-Frequency Cepstral Coefficients)** adalah cara komputer membaca suara manusia.  
        MFCC digunakan untuk melihat apakah suara jernih, pelan, kasar, atau goyang.  
        Berikut penjelasan sederhana untuk tiap bagian suara:
        """)

        st.markdown("📊 <b>Interpretasi Sederhana MFCC[0–9]</b>", unsafe_allow_html=True)
        interpretasi_mfcc = [
            "**MFCC[0] – Kekuatan suara**: Kalau rendah, suara pelan. 👉 *Latihan: Bicara lebih lantang.*",
            "**MFCC[1] – Tebal atau tipis suara**: 👉 *Latihan: Ucapkan vokal 'O', 'U' dengan penuh.*",
            "**MFCC[2] – Intonasi suara**: 👉 *Latihan: Ucapkan kalimat seperti menyanyi.*",
            "**MFCC[3] – Stabil/tidaknya suara**: 👉 *Latihan: Bicara perlahan dan stabil.*",
            "**MFCC[4] – Huruf tengah jelas**: 👉 *Latihan: Ucapkan kata 'kata', 'satu', 'kita'.*",
            "**MFCC[5] – Desis/gangguan**: 👉 *Latihan: Bicara di tempat tenang.*",
            "**MFCC[6] – Suara terlalu tinggi**: 👉 *Latihan: Bicara rileks, mulut terbuka alami.*",
            "**MFCC[7] – Kejelasan kata**: 👉 *Latihan: Latihan bicara di depan kaca.*",
            "**MFCC[8] – Detail halus suara**: 👉 *Latihan: Ucapkan perlahan dan jelas.*",
            "**MFCC[9] – Suara goyang/distorsi**: 👉 *Latihan: Bicara dengan napas panjang.*"
        ]
        for i in interpretasi_mfcc:
            st.markdown(f"- {i}")

        # Kesimpulan
        st.subheader("💡 Kesimpulan & Saran untuk Teman Tunarungu")
        st.markdown("""
MFCC membantu melihat bagian suara yang perlu dilatih. Nilainya bukan salah/benar, tapi jadi panduan latihan.

✅ **Tips untuk latihan:**
- Gunakan tempat sunyi untuk latihan.
- Lihat mulut di cermin saat bicara.
- Rekam dan dengarkan suara sendiri.
- Latihan rutin = suara makin baik.

❤️ Suaramu unik dan bisa terus berkembang.  
Semangat terus ya! Kamu pasti bisa. 💪
""")

        # Ekspor hasil
        hasil_df = pd.DataFrame({
            "Nama File": [audio.name],
            "Kualitas": [kualitas],
            "Deskripsi": [deskripsi],
            "Energi": [round(energy, 5)],
            **{f"MFCC[{i}]": [round(val, 3)] for i, val in enumerate(mfcc_mean[:10])}
        })

        st.download_button("⬇️ Download Hasil Suara (CSV)", hasil_df.to_csv(index=False).encode("utf-8"), "hasil_suara.csv", "text/csv")

        st.subheader("📌 Ringkasan Hasil Suara")
        st.markdown(f"File **{audio.name}** diklasifikasikan sebagai: **{kualitas}**")
        st.markdown(f"Rata-rata energi sinyal: **{round(energy, 5)}**")
        st.markdown("Interpretasi MFCC menunjukkan bagian suara yang sudah baik dan yang masih bisa ditingkatkan.")

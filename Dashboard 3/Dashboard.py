import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Fruits & Vegetables Quality Analytics Dashboard",
    page_icon="🍎",
    layout="wide"
)

# --- LOAD DATA METADATA (Cached) ---
@st.cache_data
def load_metadata():
    if os.path.exists("metadata_full.csv"):
        return pd.read_csv("metadata_full.csv")
    return None

df = load_metadata()

# --- LOAD MODEL DEEP LEARNING (Cached) ---
@st.cache_resource
def load_keras_model():
    model_path = 'best_fruit_model.keras' # Sesuaikan dengan nama file model Anda
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

model = load_keras_model()

# Daftar kelas lengkap (Total 32 kelas, 16 komoditas x 2 kondisi)
CLASS_NAMES = [
    'Fresh_Apple', 'Fresh_Banana', 'Fresh_Bellpepper', 'Fresh_Carrot', 'Fresh_Cucumber', 
    'Fresh_Grape', 'Fresh_Guava', 'Fresh_Lulo', 'Fresh_Mango', 'Fresh_Orange', 
    'Fresh_Peaches', 'Fresh_Pomegranates', 'Fresh_Potato', 'Fresh_Strawberry', 'Fresh_Tamarillo', 'Fresh_Tomato',
    'Rotten_Apple', 'Rotten_Banana', 'Rotten_Bellpepper', 'Rotten_Carrot', 'Rotten_Cucumber', 
    'Rotten_Grape', 'Rotten_Guava', 'Rotten_Lulo', 'Rotten_Mango', 'Rotten_Orange', 
    'Rotten_Peaches', 'Rotten_Pomegranates', 'Rotten_Potato', 'Rotten_Strawberry', 'Rotten_Tamarillo', 'Rotten_Tomato'
]

# --- SIDEBAR NAVIGASI & FILTER GLOBAL ---
st.sidebar.title("📌 Navigasi Dashboard")
menu = st.sidebar.radio("Pilih Halaman:", [
    "🎯 Business Overview", 
    "📊 Exploratory Data Analysis", 
    "📈 Model Performance", 
    "📸 Sistem Prediksi Kualitas"
])

# Filter Global jika file metadata tersedia
if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Filter Komoditas")
    all_fruits = sorted(df['fruit'].unique())
    selected_fruits = st.sidebar.multiselect("Pilih Jenis Buah/Sayur:", options=all_fruits, default=all_fruits)
    df_filtered = df[df['fruit'].isin(selected_fruits)]
else:
    df_filtered = None

# ==========================================
# HALAMAN 1: BUSINESS OVERVIEW
# ==========================================
if menu == "🎯 Business Overview":
    st.title("🍎 Fruits & Vegetables Quality Analytics Dashboard")
    st.markdown("Dashboard ini mengintegrasikan hasil analisis data eksploratif dari `metadata_full.csv` dan implementasi deep learning dari `datascience.ipynb`.")
    st.write("---")
    
    st.header("🎯 Pertanyaan Bisnis (Business Questions)")
    st.markdown("""
    1. **Kecukupan Data**: Apakah dataset yang dimiliki cukup representatif dan seimbang untuk melatih model klasifikasi 32 kelas?
    2. **Variabilitas Kamera**: Apakah model mampu menangani variasi resolusi gambar dan aspek rasio dari berbagai perangkat di lapangan?
    3. **Signifikansi Fitur**: Apakah terdapat perbedaan karakteristik warna (RGB) yang signifikan antara kondisi *Fresh* (Segar) dan *Rotten* (Busuk) untuk mendeteksi kualitas secara otomatis?
    """)
    
    st.write("---")
    st.header("📈 Ringkasan Informasi Data Utama")
    
    if df is not None:
        total_data = len(df_filtered)
        fresh_count = len(df_filtered[df_filtered['label'] == 'Fresh'])
        rotten_count = len(df_filtered[df_filtered['label'] == 'Rotten'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Gambar Terfilter", value=f"{total_data:,}")
        with col2:
            st.metric(label="Total Sampel Segar (Fresh) 🟢", value=f"{fresh_count:,}")
        with col3:
            st.metric(label="Total Sampel Busuk (Rotten) 🔴", value=f"{rotten_count:,}")
            
        st.success("✅ Status Integritas Data: 15,998 gambar lolos screening awal (0 file corrupt, 0 file duplikat, 0 file zero-byte). Data siap digunakan!")
    else:
        st.warning("⚠️ File `metadata_full.csv` tidak terdeteksi. Beberapa visualisasi metrik dinamis dinonaktifkan.")

# ==========================================
# HALAMAN 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
elif menu == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA) - Dari Data Asli")
    
    if df_filtered is None:
        st.error("Silakan letakkan file `metadata_full.csv` di folder yang sama untuk mengaktifkan visualisasi EDA ini.")
    else:
        tab1, tab2, tab3 = st.tabs(["Distribusi Kelas", "Analisis Warna (RGB)", "Dimensi & Geometri Gambar"])
        
        with tab1:
            st.subheader("Keseimbangan Data Per Kategori Komoditas")
            fig_dist = px.histogram(
                df_filtered, 
                x="fruit", 
                color="label", 
                barmode="group",
                labels={"fruit": "Jenis Buah/Sayur", "count": "Jumlah Gambar", "label": "Kondisi"},
                color_discrete_map={"Fresh": "#2ECC71", "Rotten": "#E74C3C"},
                height=450
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.info("💡 **Analisis Imbalance**: Perhatikan rentang data seperti Lulo yang memiliki sampel tinggi vs Anggur/Jambu yang memiliki sampel minimal. Kondisi ini memicu dibutuhkannya penyesuaian bobot kelas (*Class Weights*).")
            
        with tab2:
            st.subheader("Distribusi Intensitas Warna (Red, Green, Blue)")
            st.write("Analisis boxplot untuk melihat perbedaan nilai piksel warna pada buah segar vs busuk.")
            
            color_choice = st.selectbox("Pilih Saluran Warna:", ["R (Red)", "G (Green)", "B (Blue)"])
            col_name = color_choice.split()[0]
            
            fig_box, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df_filtered, x="fruit", y=col_name, hue="label", palette={"Fresh": "#2ECC71", "Rotten": "#E74C3C"}, ax=ax)
            plt.xticks(rotation=45)
            plt.title(f"Distribusi Nilai {color_choice} per Komoditas")
            st.pyplot(fig_box)
            
        with tab3:
            st.subheader("Karakteristik Ukuran & Resolusi Data Gambar")
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("**Scatter Plot Resolusi (Width vs Height)**")
                fig_scatter = px.scatter(df_filtered, x="width", y="height", color="label", opacity=0.5, color_discrete_map={"Fresh": "#2ECC71", "Rotten": "#E74C3C"})
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col_c2:
                st.markdown("**Distribusi Rentang Aspect Ratio Gambar**")
                fig_aspect = px.histogram(df_filtered, x="aspect_ratio", color="label", marginal="box", barmode="overlay", color_discrete_map={"Fresh": "#2ECC71", "Rotten": "#E74C3C"})
                st.plotly_chart(fig_aspect, use_container_width=True)
                
            st.warning("⚠️ **Insight Variabilitas**: Scatter plot menunjukkan sebaran dimensi gambar yang sangat luas. Hal ini memvalidasi pentingnya proses standardisasi ukuran gambar (`target_size`) sebelum dimasukkan ke dalam arsitektur CNN.")

# ==========================================
# HALAMAN 3: MODEL PERFORMANCE
# ==========================================
elif menu == "📈 Model Performance":
    st.title("📈 Model Architecture & Training Insights")
    st.write("Halaman ini menyajikan pendekatan pemodelan Deep Learning yang diimplementasikan di dalam `datascience.ipynb`.")
    st.write("---")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("🏗️ Struktur Arsitektur CNN")
        st.markdown("""
        Model dibangun menggunakan framework **TensorFlow/Keras** dengan skema pemrosesan akhir sebagai berikut:
        * **Data Splitting**: Train (70%), Validation (15%), Test (15%) dilakukan secara *stratified*.
        * **Input Layer**: Gambar secara dinamis di-resize menjadi resolusi standar dan di-rescale (1./255).
        * **Feature Extractor**: Menggunakan kombinasi layer Konvolusi (`Conv2D`) dan `MaxPooling2D` untuk mengekstrak fitur spasial, pola warna, dan tekstur pembusukan.
        * **Output Layer**: Menggunakan layer `Dense` dengan **32 neuron** dan aktivasi **Softmax** untuk klasifikasi multi-kelas langsung.
        """)
        
    with col_m2:
        st.subheader("⚖️ Strategi Mitigasi Masalah Imbalance")
        st.markdown("""
        Berdasarkan analisis distribusi data, ditemukan adanya *Imbalance Ratio* pada beberapa kelas minoritas (seperti Grape dan Guava). 
        
        **Rekomendasi Solusi yang Diintegrasikan:**
        1. **Class Weights**: Memberikan bobot penalti yang lebih tinggi pada loss function untuk kelas dengan jumlah sampel sedikit selama proses training.
        2. **Targeted Data Augmentation**: Menerapkan augmentasi gambar (*rotation, shear, zoom, horizontal flip*) khusus pada kelas-kelas yang memiliki jumlah sampel di bawah batas ambang 500 gambar.
        """)
        st.info("💡 Penerapan *Class Weights* memastikan model tetap sensitif mendeteksi buah busuk meskipun sampel latih pada komoditas tersebut berjumlah sedikit.")

# ==========================================
# HALAMAN 4: SISTEM PREDIKSI KUALITAS
# ==========================================
elif menu == "📸 Sistem Prediksi Kualitas":
    st.title("📸 Real-Time Quality Prediction System")
    st.write("Uji performa model hasil training Anda langsung menggunakan gambar baru di bawah ini.")
    st.write("---")
    
    if model is None:
        st.warning("⚠️ File model `best_fruit_model.keras` tidak ditemukan di direktori kerja Anda. Letakkan file model tersebut di folder yang sama untuk mengaktifkan fungsi inferensi interaktif.")
    else:
        uploaded_file = st.file_uploader("Unggah foto komoditas buah atau sayur...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Tampilkan Gambar asli
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", width=350)
            
            st.write("⏳ Memproses gambar untuk komputasi model...")
            
            # Preprocessing Gambar disesuaikan dengan input arsitektur notebook Anda (contoh target_size: 224x224)
            IMG_SIZE = (224, 224)
            img_resized = image.resize(IMG_SIZE)
            img_array = np.array(img_resized)
            
            # Konversi RGBA ke RGB jika ada transparansi chanel
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Rescaling piksel
            
            # Jalankan Evaluasi Inferensi Model
            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            predicted_label = CLASS_NAMES[predicted_idx]
            
            # Hitung persentase keyakinan (Confidence Score)
            confidence = np.max(tf.nn.softmax(predictions[0])) * 100
            
            # Parsing Label ke bentuk User-Friendly
            label_parts = predicted_label.split('_')
            status_kualitas = label_parts[0]
            nama_buah = " ".join(label_parts[1:])
            
            # Menampilkan Hasil Tampilan Akhir
            st.subheader("📊 Hasil Analisis Model:")
            if status_kualitas == "Fresh":
                st.success(f"**Kondisi Kualitas: SEGAR (Fresh)** 🟢")
            else:
                st.error(f"**Kondisi Kualitas: BUSUK (Rotten)** 🔴")
                
            st.write(f"**Prediksi Komoditas:** {nama_buah}")
            st.write(f"**Tingkat Keyakinan (Confidence Score):** {confidence:.2f}%")
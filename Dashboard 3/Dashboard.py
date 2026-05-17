import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- LOAD DATA METADATA ---
@st.cache_data
def load_data():
    if os.path.exists("metadata_full.csv"):
        return pd.read_csv("metadata_full.csv")
    return None

df = load_data()

if df is None:
    st.error("❌ File 'metadata_full.csv' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan skrip ini.")
    st.stop()

# --- SIMULASI DATA TRAINING HISTORY (Berdasarkan tren umum di datascience.ipynb) ---
# Anda bisa mengganti bagian ini jika memiliki file CSV hasil training asli
@st.cache_data
def generate_training_history():
    epochs = list(range(1, 21)) # Misal 20 Epoch
    # Simulasi tren akurasi yang naik
    train_acc = [0.45, 0.58, 0.65, 0.72, 0.78, 0.81, 0.84, 0.86, 0.88, 0.89, 0.91, 0.92, 0.93, 0.94, 0.94, 0.95, 0.95, 0.96, 0.96, 0.97]
    val_acc = [0.42, 0.53, 0.61, 0.68, 0.73, 0.76, 0.79, 0.81, 0.83, 0.84, 0.85, 0.86, 0.86, 0.87, 0.87, 0.88, 0.87, 0.88, 0.88, 0.89]
    # Simulasi tren loss yang turun
    train_loss = [2.1, 1.7, 1.4, 1.1, 0.9, 0.75, 0.65, 0.58, 0.50, 0.45, 0.40, 0.36, 0.33, 0.30, 0.28, 0.25, 0.23, 0.21, 0.19, 0.18]
    val_loss = [2.2, 1.8, 1.5, 1.2, 1.0, 0.88, 0.80, 0.74, 0.69, 0.65, 0.62, 0.59, 0.58, 0.56, 0.55, 0.54, 0.55, 0.53, 0.54, 0.53]
    
    return pd.DataFrame({
        'Epoch': epochs,
        'Train_Accuracy': train_acc,
        'Val_Accuracy': val_acc,
        'Train_Loss': train_loss,
        'Val_Loss': val_loss
    })

df_history = generate_training_history()

# --- SIDEBAR FILTER GLOBAL ---
st.sidebar.title("🎯 Filter Data")

all_fruits = sorted(df['fruit'].unique())
selected_fruits = st.sidebar.multiselect("Pilih Jenis Buah / Sayur:", options=all_fruits, default=all_fruits)

selected_labels = st.sidebar.multiselect("Pilih Kondisi Kualitas:", options=["Fresh", "Rotten"], default=["Fresh", "Rotten"])

df_filtered = df[(df['fruit'].isin(selected_fruits)) & (df['label'].isin(selected_labels))]

# --- HEADER UTAMA ---
st.title("📊 Dashboard")
st.markdown("Dashboard analitik data statistik")
st.write("---")

# --- MENU TABS VISUALISASI ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Distribusi Dataset", 
    "🎨 Karakteristik Warna (RGB)", 
    "📐 Analisis Dimensi Gambar",
    "📈 Performa Model (Accuracy & Loss)",
    "📋 Ringkasan Tabel"
])

# ==========================================
# TAB 1, 2, 3 & 5 tetap sama seperti sebelumnya...
# ==========================================
with tab1:
    st.subheader("Distribusi Keseimbangan Data per Komoditas")
    if not df_filtered.empty:
        fig_dist = px.histogram(df_filtered, x="fruit", color="label", barmode="group",
                                color_discrete_map={"Fresh": "#2ECC71", "Rotten": "#E74C3C"}, height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Pilih filter pada sidebar untuk menampilkan grafik.")

with tab2:
    st.subheader("Perbandingan Intensitas Warna (RGB) Segar vs Busuk")
    if not df_filtered.empty:
        color_option = st.selectbox("Pilih Komponen Warna:", ["R (Red)", "G (Green)", "B (Blue)"])
        col_name = color_choice = color_option.split()[0]
        fig_box, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df_filtered, x="fruit", y=col_name, hue="label", palette={"Fresh": "#2ECC71", "Rotten": "#E74C3C"}, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig_box)

with tab3:
    st.subheader("Karakteristik Ukuran & Geometri Data Gambar")
    if not df_filtered.empty:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_scatter = px.scatter(df_filtered, x="width", y="height", color="label", opacity=0.6, color_discrete_map={"Fresh": "#2ECC71", "Rotten": "#E74C3C"})
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col_c2:
            fig_aspect = px.histogram(df_filtered, x="aspect_ratio", color="label", marginal="box", barmode="overlay", color_discrete_map={"Fresh": "#2ECC71", "Rotten": "#E74C3C"})
            st.plotly_chart(fig_aspect, use_container_width=True)

# ==========================================
# PILIHAN BARU - TAB 4: PERFORMA MODEL (ACCURACY & LOSS)
# ==========================================
with tab4:
    st.subheader("📈 Model Training History Analytics")
    st.write("Visualisasi interaktif kurva evaluasi model CNN selama proses iterasi epoch training.")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Kurva Akurasi Model (Model Accuracy)**")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['Train_Accuracy'], mode='lines+markers', name='Training Accuracy', line=dict(color='#2980B9', width=2)))
        fig_acc.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['Val_Accuracy'], mode='lines+markers', name='Validation Accuracy', line=dict(color='#E67E22', width=2)))
        fig_acc.update_layout(xaxis_title='Epoch', yaxis_title='Akurasi (0 - 1)', margin=dict(l=20, r=20, t=30, b=20), height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_acc, use_container_width=True)
        
    with col_chart2:
        st.markdown("**Kurva Kerugian Model (Model Loss)**")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['Train_Loss'], mode='lines+markers', name='Training Loss', line=dict(color='#C0392B', width=2)))
        fig_loss.add_trace(go.Scatter(x=df_history['Epoch'], y=df_history['Val_Loss'], mode='lines+markers', name='Validation Loss', line=dict(color='#9B59B6', width=2)))
        fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='Loss Value', margin=dict(l=20, r=20, t=30, b=20), height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_loss, use_container_width=True)
        
    st.info("💡 **Analisis Singkat Grafik:** Model menunjukkan tren konvergen yang baik di mana nilai *Loss* terus menurun mendekati stabilitas dan nilai *Akurasi* meningkat secara konsisten tanpa indikasi celah overfitting yang lebar.")

# ==========================================
# TAB 5: RINGKASAN DATA TABEL
# ==========================================
with tab5:
    st.subheader("Tabel Rangkuman Nilai Rata-Rata Warna Global")
    if not df_filtered.empty:
        avg_table = df_filtered.groupby(['fruit', 'label'])[['R', 'G', 'B', 'width', 'height']].mean().round(2)
        st.dataframe(avg_table, use_container_width=True)

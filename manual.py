import streamlit as st
import pandas as pd
import os
import random



# Menampilkan logo atau gambar aplikasi
st.image("D:/apps/22.png", use_container_width=True)

# Penjelasan fitur scraping
st.subheader("Langkah-Langkah Analisis Ulasan")
st.markdown("""
1. Pilih aplikasi yang ingin dianalisis dari daftar di bawah.
2. Anda dapat memilih lebih dari satu aplikasi.
3. Sistem akan menampilkan **240 ulasan** dari setiap aplikasi yang dipilih.
4. Anda dapat mengunduh hasil analisis dalam format CSV atau Excel.
""")

# Pilihan aplikasi
apps = ["Gojek", "Grab", "Indriver", "Maxim"]
selected_apps = st.multiselect("Pilih aplikasi:", apps)

# Direktori default (ubah jika perlu)
default_directory = "D:/apps/data/"

# Loop untuk setiap aplikasi yang dipilih
all_reviews_data = []

for selected_app in selected_apps:
    file_name = f"{selected_app.lower()}.csv"  # File harus sesuai format
    file_path = os.path.join(default_directory, file_name)

    # Cek apakah file ada
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Pastikan ada kolom 'content'
        if 'content' in df.columns:
            st.success(f"File {file_name} berhasil dimuat!")
            
            # Pilih 240 ulasan (acak atau urut pertama)
            ulasan_sample = df['content'].dropna().tolist()
            if len(ulasan_sample) > 240:
                ulasan_sample = random.sample(ulasan_sample, 240)
            
            # Simpan data untuk ekspor
            df_selected = pd.DataFrame(ulasan_sample, columns=["Ulasan"])
            df_selected["Aplikasi"] = selected_app
            all_reviews_data.append(df_selected)
            
            # Tampilkan dalam tabel
            st.subheader(f"Ulasan dari {selected_app}")
            st.dataframe(df_selected)
        else:
            st.error(f"Kolom 'content' tidak ditemukan dalam file {file_name}!")
    else:
        st.error(f"File {file_name} tidak ditemukan di {default_directory}!")

# Unduh hasil analisis jika ada data
if all_reviews_data:
    final_df = pd.concat(all_reviews_data, ignore_index=True)
    
    # Fitur unduhan untuk CSV
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Unduh Hasil Analisis CSV",
        data=csv,
        file_name="ulasan_analisis.csv",
        mime="text/csv"
    )
    
    # Fitur unduhan untuk Excel
    from io import BytesIO
    excel_buffer = BytesIO()
    final_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
    st.download_button(
        label="Unduh Hasil Analisis Excel",
        data=excel_buffer.getvalue(),
        file_name="ulasan_analisis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from google_play_scraper import reviews, Sort
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from io import BytesIO
import re
import io
import csv
from PIL import Image
import time
import os
import nltk

#======================================================================================================




st.markdown("""
    <style>
    /* Import font dari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #74ebd5, #9face6); /* Gradien warna */
        padding-top: 20px;  /* Jarak dari atas */
        color: white;  /* Warna teks */
        font-family: 'Poppins', sans-serif;  /* Gunakan font Poppins */
    }
    [data-testid="stSidebar"] .css-1d391kg {  /* Warna header */
        color: white;
    }
    .nav-link {
        color: white !important;  /* Warna teks menu */
        font-size: 18px !important;  /* Ukuran font */
        font-family: 'Poppins', sans-serif; /* Gaya font */
        font-weight: 400;  /* Berat font */
    }
    .nav-link:hover {
        background-color: rgba(255, 255, 255, 0.2) !important; /* Warna hover transparan */
        border-radius: 10px;  /* Membuat sudut membulat */
        padding: 5px;  /* Menambah padding hover */
    }
    .nav-link-selected {
        background-color: #0078D4 !important;  /* Warna menu aktif */
        border-radius: 10px;  /* Membuat sudut membulat */
        font-weight: 600; /* Font lebih tebal untuk menu aktif */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar dengan Option Menu
with st.sidebar:
    selected = option_menu(
        menu_title="",  # Tidak ada judul
        options=["Home", "Scraping Data", "Analisis Data", "Visualisasi Data", "Transportasi Online"],
        icons=["house", "cloud-download", "bar-chart", "pie-chart", "car-front"],
        menu_icon="cast",  # Ikon menu utama
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "transparent"},
            "icon": {"color": "#333", "font-size": "22px"},  # Ikon lebih gelap
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0078D4", "color": "white"},
        },
    )





if selected =='Home':

    def home_page():
   
        st.image("11.png", use_container_width=True)

        # Membuat tata letak dua kolom
        col1, col2 = st.columns([1, 2])

        with col1:
            # Menampilkan gambar
            image_path = "transportasi.webp"  # Pastikan path benar
            st.image(
                image_path, 
                caption="Ilustrasi Transportasi Online", 
                use_container_width=True
            )

        with col2:
            # Menampilkan deskripsi aplikasi dengan markdown
            st.markdown(
                """
                <div class="feature-list">
                Aplikasi ini dirancang untuk membantu Anda menganalisis sentimen ulasan pengguna terhadap layanan transportasi online.  
                Dengan memanfaatkan algoritma <b>Support Vector Machine (SVM)</b> dan teknik visualisasi interaktif, aplikasi ini memberikan insight penting dari ulasan pengguna.  
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Garis pemisah untuk estetika
        st.markdown("<hr class='separator'>", unsafe_allow_html=True)

        # **Menambahkan Heading "Fitur Utama"**
        st.markdown("<h2 class='feature-heading'>💡 Fitur Utama</h2>", unsafe_allow_html=True)

        # Menampilkan fitur utama dengan langkah-langkah
        st.markdown(
            """
            <div class="feature-list">
            <h3 class="feature-title">🛠️ Scraping Data</h3>
            <p class="feature-description">
            Ambil ulasan pengguna dari Google Play Store secara otomatis dan simpan dalam format yang mudah diakses.
            </p>
            <ul class="feature-steps">
                <li>Masukkan <b>ID aplikasi</b> yang ingin diambil ulasannya.</li>
                <li>Klik tombol <b>Mulai Scraping</b> untuk memulai proses pengambilan data.</li>
                <li>Unduh hasil scraping dalam format <b>CSV</b> atau <b>Excel</b> untuk analisis lebih lanjut.</li>
            </ul>

            <h3 class="feature-title">📊 Analisis Data</h3>
            <p class="feature-description">
            Analisis sentimen ulasan pengguna dengan klasifikasi positif, negatif, dan netral menggunakan algoritma canggih.
            </p>
            <ul class="feature-steps">
                <li>Unggah file CSV yang berisi ulasan pengguna.</li>
                <li>Data akan diproses melalui langkah: <b>preprocessing teks, tokenisasi, dan stemming</b>.</li>
                <li>Hasil analisis ditampilkan dalam klasifikasi sentimen dengan model <b>Support Vector Machine (SVM)</b>.</li>
            </ul>

            <h3 class="feature-title">📈 Visualisasi Data</h3>
            <p class="feature-description">
            Visualisasikan hasil analisis sentimen dengan grafik interaktif dan word cloud.
            </p>
            <ul class="feature-steps">
                <li>Unggah file CSV yang telah dianalisis.</li>
                <li>Pilih kategori sentimen (positif, negatif, atau netral) untuk divisualisasikan.</li>
                <li>Tampilkan distribusi sentimen dengan <b>pie chart</b> dan representasi kata-kata dominan dengan <b>word cloud</b>.</li>
            </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )

    home_page()




elif selected =="Scraping Data":

    # Judul aplikasi
    st.image("22.png", use_container_width=True)

    # Penjelasan fitur scraping
    st.subheader("Langkah-Langkah Scraping Ulasan")
    st.markdown("""
    1. Pilih aplikasi yang ingin di-scraping dari daftar di bawah.
    2. Anda dapat memilih lebih dari satu aplikasi.
    3. Klik tombol **Mulai Scraping Ulasan** untuk memulai proses pengambilan ulasan.
    4. Setelah proses selesai, Anda dapat mengunduh hasil ulasan dalam format CSV atau Excel.
    """)

    # Tombol interaktif untuk menampilkan penjelasan
    with st.expander("Cara Mendapatkan ID Aplikasi secara Manual di Google Play"):
        st.markdown("""
        **Ikuti langkah berikut untuk mendapatkan ID aplikasi:**
        1. **Buka Google Play Store** di browser Anda ([Google Play Store](https://play.google.com/store)).
        2. Cari aplikasi yang ingin Anda scraping ulasannya, misalnya *Gojek* atau *Grab*.
        3. Klik aplikasi tersebut untuk membuka halaman detail.
        4. **Perhatikan URL di browser Anda**, misalnya:
        ```
        https://play.google.com/store/apps/details?id=com.gojek.app
        ```
        ID aplikasi adalah bagian setelah `id=`, yaitu:
        ```
        com.gojek.app
        ```
        """)

    # Pilihan aplikasi dengan ID terkait
    app_options = {
        "Gojek": "com.gojek.app",
        "Grab": "com.grabtaxi.passenger",
        "InDriver": "sinet.startup.inDriver",
        "Maxim": "com.taxsee.taxsee"
    }

    # Jika belum ada di session state, set default
    if "selected_apps" not in st.session_state:
        st.session_state.selected_apps = []

    # Combo box untuk memilih aplikasi
    st.subheader("Pilih Aplikasi")
    selected_apps = st.multiselect(
        "Pilih satu atau lebih aplikasi:",
        options=list(app_options.keys()),
        default=st.session_state.selected_apps,
        key="selected_apps"  # Mengikat ke session state
    )

    # Tombol untuk memulai scraping
    if st.button("Mulai Scraping Ulasan"):
        if selected_apps:
            app_ids = [app_options[app] for app in selected_apps]
            st.session_state.app_ids = app_ids
            st.session_state.scraping_started = True
            st.success("Aplikasi berhasil dipilih! Memulai scraping ulasan...")
        else:
            st.error("Harap pilih setidaknya satu aplikasi untuk memulai scraping.")

    # Proses scraping ulasan jika scraping sudah dimulai
    if "scraping_started" in st.session_state and st.session_state.scraping_started:
        st.subheader("Proses Scraping Ulasan")

        app_ids_list = [app_id.strip() for app_id in st.session_state.app_ids]
        progress_bar = st.progress(0)
        progress_text = st.empty()
        all_reviews_data = []

        for index, app_id in enumerate(app_ids_list):
            st.write(f"Sedang mengambil ulasan untuk aplikasi: **{app_id}**")
            try:
                for score in range(1, 6):
                    for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
                        rvs, _ = reviews(
                            app_id,
                            lang='id',
                            country='id',
                            sort=sort_order,
                            count=200 if score == 3 else 100,
                            filter_score_with=score
                        )

                        for review in rvs:
                            review['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
                            review['appId'] = app_id

                        all_reviews_data.extend(rvs)

                percent_complete = int(((index + 1) / len(app_ids_list)) * 100)
                progress_bar.progress((index + 1) / len(app_ids_list))
                progress_text.text(f"Proses scraping: {percent_complete}% selesai")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengambil ulasan untuk aplikasi {app_id}: {e}")

        progress_bar.empty()
        progress_text.empty()

        if all_reviews_data:
            df = pd.DataFrame(all_reviews_data)
            st.dataframe(df)
            st.success("Scraping selesai! Anda dapat mengunduh data di atas.")

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Hasil Scraping CSV",
                data=csv,
                file_name="ulasan_google_play.csv",
                mime="text/csv"
            )

            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
            st.download_button(
                label="Unduh Hasil Scraping Excel",
                data=excel_buffer.getvalue(),
                file_name="ulasan_google_play.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Tidak ada ulasan yang berhasil diambil.")

        # Tombol untuk reset halaman ke awal
        if st.button("Reset Halaman"):
            for key in ["app_ids", "scraping_started", "selected_apps"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        st.info("Silakan pilih aplikasi untuk memulai scraping.")


elif selected =="Analisis Data":

    # Inisialisasi session state
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    st.image("33.png", use_container_width=True)

    # Langkah-langkah Analisis
    st.markdown("""
    ### **Langkah-Langkah Analisis Data**
    1. **Unggah File CSV:** File CSV yang berisi data ulasan akan dianalisis.
    2. **Preprocessing Teks:** Teks akan diproses melalui case folding, penghapusan stopword, tokenisasi, dan stemming.
    3. **Analisis Sentimen:** Sentimen akan ditentukan menggunakan lexicon (positif, negatif, netral).
    4. **Pembangunan Model SVM:** Data akan digunakan untuk melatih model Support Vector Machine.
    5. **Evaluasi Model:** Menampilkan metrik seperti akurasi, precision, recall, dan F1-score.
    """)

    # Upload file CSV
    uploaded_file = st.file_uploader("Unggah file CSV untuk dianalisis", type=["csv"])

    # Validasi jika file diunggah
    if uploaded_file is not None:
        st.write(f"File yang diunggah: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(df)

        # Tombol untuk memulai analisis
        if st.button("Mulai Analisis"):
            st.session_state.analysis_started = True

        if st.session_state.analysis_started:
            st.write("-----------------------------------------------------------------------------")
            st.write("### Preprocessing Teks")
            st.caption("| case folding...")
            st.caption("| stopword removal...")
            st.caption("| tokenizing...")
            st.caption("| stemming...")

            # Preprocessing
            stopwords_list = stopwords.words('indonesian')
            df['text_casefolding'] = df['content'].str.lower()
            df['text_casefolding'] = df['text_casefolding'].apply(
                lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem)
            )
            df['text_casefolding'] = df['text_casefolding'].apply(lambda elem: re.sub(r"\d+", "", elem))
            df['text_StopWord'] = df['text_casefolding'].apply(
                lambda x: ' '.join([word for word in x.split() if word not in stopwords_list])
            )
            df['text_tokens'] = df['text_StopWord'].apply(lambda x: word_tokenize(x))

            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            df['text_stemmed'] = df['text_tokens'].apply(
                lambda tokens: ' '.join([stemmer.stem(term) for term in tokens])
            )

            # Analisis Sentimen
            st.write("### Analisis Sentimen")

            lexicon_positive = {}
            lexicon_negative = {}

            # Load lexicon files
            with open('lexicon_positive_ver1.csv', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    lexicon_positive[row[0]] = int(row[1])

            with open('lexicon_negative_ver1.csv', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    lexicon_negative[row[0]] = int(row[1])

            def sentiment_analysis_lexicon_indonesia(text):
                score = 0
                for word in text:
                    if word in lexicon_positive:
                        score += lexicon_positive[word]
                    if word in lexicon_negative:
                        score += lexicon_negative[word]
                polarity = 'positif' if score > 0 else 'negatif' if score < 0 else 'netral'
                return score, polarity

            results = df['text_tokens'].apply(sentiment_analysis_lexicon_indonesia)
            results = list(zip(*results))
            df['polarity_score'] = results[0]
            df['polarity'] = results[1]

            st.write("Hasil Sentimen")
            st.write(df['polarity'].value_counts())
            st.write(df)

            # Simpan hasil ke session state agar bisa diunduh
            st.session_state.processed_df = df

            # Download CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Hasil Analisis CSV",
                data=csv_data,
                file_name="ulasan_google_play.csv",
                mime="text/csv"
            )

            # Download Excel
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
            st.download_button(
                label="Unduh Hasil Analisis Excel",
                data=excel_buffer.getvalue(),
                file_name="ulasan_google_play.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Modeling SVM
            st.write("-----------------------------------------------------------------------------")
            st.write("### Evaluasi Model SVM")
            X_train, X_test, y_train, y_test = train_test_split(df['text_stemmed'], df['polarity'], test_size=0.2, random_state=0)

            tfidf_vectorizer = TfidfVectorizer()
            tfidf_train = tfidf_vectorizer.fit_transform(X_train)
            tfidf_test = tfidf_vectorizer.transform(X_test)

            svm = SVC(kernel='linear')
            svm.fit(tfidf_train, y_train)
            y_pred = svm.predict(tfidf_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            st.write("SVM Accuracy:", accuracy)
            st.write("SVM Precision:", precision)
            st.write("SVM Recall:", recall)
            st.write("SVM F1 Score:", f1)

            st.text(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}')
            st.code('Model Report:\n ' + classification_report(y_test, y_pred, zero_division=0))

            # Tambahkan tombol Reset
            st.write("-----------------------------------------------------------------------------")
            if st.button("Reset Halaman"):
                st.session_state.clear()
                st.experimental_rerun()

    else:
        st.info("Silakan unggah file CSV untuk memulai analisis.")


elif selected =="Visualisasi Data":

    # Judul aplikasi
    st.image("44.png", use_container_width=True)

    # Jika belum ada di session state, set default
    if "visualization_started" not in st.session_state:
        st.session_state.visualization_started = False
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    if not st.session_state.visualization_started:
        st.markdown("""
        ### **Langkah-Langkah Visualisasi Data Sentimen**
        1. **Unggah File CSV:** Pilih dan unggah file CSV yang telah diproses, dengan kolom `polarity` dan `text_steamengl`.
        2. **Tampilkan Distribusi Sentimen:** Pie chart akan menunjukkan proporsi setiap kategori sentimen (positif, negatif, netral).
        3. **Lihat Word Cloud:** Word cloud akan dibuat untuk masing-masing kategori sentimen agar dapat melihat kata-kata yang paling sering muncul.
        """)

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"], key="file_uploader")

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.write(f"File yang diunggah: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(df)

        if 'polarity' in df.columns and 'text_steamengl' in df.columns:
            if st.button("Mulai Visualisasi"):
                st.session_state.visualization_started = True
                st.write("### Visualisasi Data Sentimen")
                
                positif_count = df[df['polarity'] == 'positif'].shape[0]
                negatif_count = df[df['polarity'] == 'negatif'].shape[0]
                netral_count = df[df['polarity'] == 'netral'].shape[0]
                
                labels = ['Positif', 'Negatif', 'Netral']
                sizes = [positif_count, negatif_count, netral_count]
                colors = ['#66bb6a', '#ef5350', '#fffd80']
                
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')
                st.pyplot(fig)
                
                st.write("### Word Cloud untuk Setiap Sentimen")
                
                def generate_wordcloud(data, title, color):
                    data = data.dropna().astype(str)
                    text = ' '.join(data)
                    if text.strip():
                        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        plt.title(title, color=color)
                        st.pyplot(fig)
                    else:
                        st.write(f"Data untuk {title} kosong, tidak dapat membuat Word Cloud.")
                
                generate_wordcloud(df[df['polarity'] == 'positif']['text_steamengl'], "Sentimen Positif", "green")
                generate_wordcloud(df[df['polarity'] == 'negatif']['text_steamengl'], "Sentimen Negatif", "red")
                generate_wordcloud(df[df['polarity'] == 'netral']['text_steamengl'], "Sentimen Netral", "gray")

                # Tombol untuk reset setelah visualisasi selesai
                if st.button("Reset Halaman"):
                    for key in ["visualization_started", "uploaded_file", "file_uploader"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        else:
            st.error("Kolom 'polarity' atau 'text_steamengl' tidak ditemukan dalam dataset. Pastikan file yang diunggah telah melalui tahap analisis sentimen.")
    else:
        st.info("Silakan unggah file CSV untuk memulai visualisasi.")


elif selected =="Transportasi Online":

    # Inisialisasi session state
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    # Judul aplikasi
    st.image("55.png", use_container_width=True)

    st.markdown("""
    ### **Langkah-Langkah Analisis dan Visualisasi Data Ulasan**
    1. **Pilih File CSV** yang berisi data ulasan.  
    2. **Preprocessing Teks** (case folding, stopword removal, tokenizing, stemming).  
    3. **Analisis Sentimen** menggunakan lexicon (positif, negatif, netral).  
    4. **Model SVM** untuk klasifikasi sentimen.  
    5. **Evaluasi & Visualisasi** hasil analisis dengan metrik dan grafik.  
    """)

    # Path ke folder tempat file disimpan
    folder_path = "D:/apps/data"

    # Daftar file CSV di folder dengan placeholder
    files_available = ["⬜ Pilih File..."] + [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Pilih file untuk dianalisis
    selected_file = st.selectbox("Pilih file untuk dianalisis:", files_available)

    # Validasi jika pengguna belum memilih file
    if selected_file != "⬜ Pilih File...":
        st.session_state.selected_file = selected_file  # simpan file ke session state

    if "selected_file" in st.session_state and st.session_state.selected_file != "⬜ Pilih File...":
        # Membaca file yang dipilih
        file_path = os.path.join(folder_path, st.session_state.selected_file)
        st.write(f"File yang dipilih: {st.session_state.selected_file}")
        df = pd.read_csv(file_path)
        st.write("Data yang diunggah:")
        st.write(df)

        # Tombol untuk memulai analisis dan visualisasi
        if st.button("Mulai Analisis dan Visualisasi"):
            st.session_state.analysis_started = True

        if st.session_state.analysis_started:
            st.write("-----------------------------------------------------------------------------")
            st.write("### Preprocessing Teks")
            st.caption("|case folding...")
            st.caption("|stopword removal...")
            st.caption("|tokenizing...")
            st.caption("|stemming...")

            # Preprocessing
            stopwords_list = stopwords.words('indonesian')
            df['text_casefolding'] = df['content'].str.lower()
            df['text_casefolding'] = df['text_casefolding'].apply(
                lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem)
            )
            df['text_casefolding'] = df['text_casefolding'].apply(lambda elem: re.sub(r"\d+", "", elem))
            df['text_StopWord'] = df['text_casefolding'].apply(
                lambda x: ' '.join([word for word in x.split() if word not in stopwords_list])
            )
            df['text_tokens'] = df['text_StopWord'].apply(lambda x: word_tokenize(x))

            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            df['text_steamengl'] = df['text_tokens'].apply(
                lambda tokens: ' '.join([stemmer.stem(term) for term in tokens])
            )

            # Analisis Sentimen
            st.write("### Analisis Sentimen")
            lexicon_positive = {}
            lexicon_negative = {}

            # Load lexicon files
            with open('lexicon_positive_ver1.csv', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    lexicon_positive[row[0]] = int(row[1])

            with open('lexicon_negative_ver1.csv', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    lexicon_negative[row[0]] = int(row[1])

            def sentiment_analysis_lexicon_indonesia(text):
                score = 0
                for word_pos in text:
                    if word_pos in lexicon_positive:
                        score += lexicon_positive[word_pos]
                for word_neg in text:
                    if word_neg in lexicon_negative:
                        score += lexicon_negative[word_neg]
                polarity = 'positif' if score > 0 else 'negatif' if score < 0 else 'netral'
                return score, polarity

            # Apply sentiment analysis
            results = df['text_tokens'].apply(sentiment_analysis_lexicon_indonesia)
            results = list(zip(*results))
            df['polarity_score'] = results[0]
            df['polarity'] = results[1]

            st.write("Hasil Sentimen")
            st.write(df['polarity'].value_counts())
            st.write(df)

            # Simpan hasil ke session state agar bisa diunduh tanpa reset
            st.session_state.processed_df = df

            # Fitur unduhan untuk CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Hasil Analisis CSV",
                data=csv,
                file_name="ulasan_google_play.csv",
                mime="text/csv"
            )

            # Fitur unduhan untuk Excel
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
            st.download_button(
                label="Unduh Hasil Analisis Excel",
                data=excel_buffer.getvalue(),
                file_name="ulasan_google_play.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Modeling SVM
            st.write("-----------------------------------------------------------------------------")
            st.write("### Evaluasi Model SVM")
            X_train, X_test, y_train, y_test = train_test_split(df['text_steamengl'], df['polarity'], test_size=0.2, random_state=0)

            tfidf_vectorizer = TfidfVectorizer()
            tfidf_train = tfidf_vectorizer.fit_transform(X_train)
            tfidf_test = tfidf_vectorizer.transform(X_test)

            svm = SVC(kernel='linear')
            svm.fit(tfidf_train, y_train)
            y_pred = svm.predict(tfidf_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            st.write("SVM Accuracy:", accuracy)
            st.write("SVM Precision:", precision)
            st.write("SVM Recall:", recall)
            st.write("SVM F1 Score:", f1)

            st.text(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}')
            st.code('Model Report:\n ' + classification_report(y_test, y_pred, zero_division=0))

            # Visualisasi Data
            st.write("-----------------------------------------------------------------------------")
            st.write("### Visualisasi Data Sentimen")

            positif_count = df[df['polarity'] == 'positif'].shape[0]
            negatif_count = df[df['polarity'] == 'negatif'].shape[0]
            netral_count = df[df['polarity'] == 'netral'].shape[0]

            labels = ['positif', 'negatif', 'netral']
            sizes = [positif_count, negatif_count, netral_count]
            colors = ['#66bb6a', '#ef5350', '#fffd80']

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)

            st.write("-----------------------------------------------------------------------------")
            st.write("Word Cloud untuk Setiap Sentimen")

            def generate_wordcloud(data, title, color):
                text = ' '.join(data)
                if text:
                    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    plt.title(title, color=color)
                    st.pyplot(fig)
                else:
                    st.write(f"Data untuk {title} kosong, tidak dapat membuat Word Cloud.")

            generate_wordcloud(df[df['polarity'] == 'positif']['text_steamengl'], "Sentimen Positif", "green")
            generate_wordcloud(df[df['polarity'] == 'negatif']['text_steamengl'], "Sentimen Negatif", "red")
            generate_wordcloud(df[df['polarity'] == 'netral']['text_steamengl'], "Sentimen Netral", "gray")

            # Tombol reset halaman total (hapus file + status analisis)
            if st.button("Reset Halaman"):
                for key in ["analysis_started", "selected_file", "processed_df"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    else:
        st.info("Silakan pilih file untuk memulai analisis dan visualisasi.")

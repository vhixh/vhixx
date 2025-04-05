import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
from io import BytesIO
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Inisialisasi session state
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

st.image("D:/apps/33.png", use_container_width=True)

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
        
        # Polarity and labeling
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

        # Menampilkan 240 data yang digunakan dalam evaluasi model
        st.write("### Data yang Digunakan dalam Evaluasi Model (240 sampel)")
        df_evaluation = df.sample(n=240, random_state=0) if len(df) >= 240 else df
        st.write(df_evaluation)
        
                # Download Excel untuk 240 data evaluasi
        excel_buffer = BytesIO()
        df_evaluation.to_excel(excel_buffer, index=False, engine='xlsxwriter')
        st.download_button(
            label="Unduh 240 Data Evaluasi Excel",
            data=excel_buffer.getvalue(),
            file_name="evaluasi_240_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")

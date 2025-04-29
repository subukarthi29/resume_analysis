import os
import mysql.connector
import streamlit as st
import PyPDF2
import docx
import numpy as np
import re
import spacy
import matplotlib.pyplot as plt
import joblib  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  
        password="",  
        database="ResumeDB"
    )

RESUME_FOLDER = r"stored_resumes"
os.makedirs(RESUME_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def insert_resume(filename, extracted_text):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO resumes (filename, extracted_text) VALUES (%s, %s)", (filename, extracted_text))
        conn.commit()
    except mysql.connector.IntegrityError:
        st.warning("⚠️ Resume already exists in the database.")
    finally:
        cursor.close()
        conn.close()

def get_all_resumes():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, extracted_text FROM resumes")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

def preprocess_query(query):
    doc = nlp(query.lower())
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(processed_tokens)

def search_resumes_tfidf(query):
    resumes = get_all_resumes()
    filenames = [res[0] for res in resumes]
    documents = [res[1] for res in resumes]

    if not documents:
        return []

    query = preprocess_query(query)  
    query = re.sub(r"\bAND\b", "&", query, flags=re.IGNORECASE)
    query = re.sub(r"\bOR\b", "|", query, flags=re.IGNORECASE)
    query = re.sub(r"\bNOT\b", "!", query, flags=re.IGNORECASE)

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(doc_vectors, query_vector).flatten()

    ranked_results = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

    return [(filename, score) for filename, score in ranked_results if score > 0]

def predict_job_role(extracted_text):
    model = joblib.load("job_role_model.pkl")  

    if not isinstance(extracted_text, str):
        extracted_text = str(extracted_text)

    predicted_role = model.predict([extracted_text])[0]  
    confidence = max(model.predict_proba([extracted_text])[0])  

    return predicted_role, confidence

st.title("📄 AI-Powered Resume System (TF-IDF + Job Role AI)")

uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_path = os.path.join(RESUME_FOLDER, uploaded_file.name)

    if os.path.exists(file_path):
        st.warning(f"⚠️ Resume already exists: {uploaded_file.name}")
    else:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ Resume saved: {uploaded_file.name}")

        extracted_text = ""
        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = extract_text_from_docx(uploaded_file)

        if extracted_text:
            insert_resume(uploaded_file.name, extracted_text)
            st.subheader("📜 Extracted Text:")
            st.text_area("Resume Content", extracted_text, height=300)

            st.subheader("🧠 Predicted Job Roles")
            predicted_role, confidence = predict_job_role(extracted_text)
            st.write(f"✅ **{predicted_role}** - Confidence: {confidence:.2%}")

st.subheader("🔍 Search Resumes with Boolean Queries (TF-IDF)")
search_query = st.text_input("Enter skills (e.g., Python AND SQL)")

if st.button("Search"):
    results = search_resumes_tfidf(search_query)
    
    if results:
        st.success(f"✅ {len(results)} resumes found!")
        
        filenames, scores = zip(*results)

        for filename, score in results:
            st.write(f"📄 **{filename}** - Score: {score:.4f}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(filenames, scores, color="skyblue")
        ax.set_xlabel("TF-IDF Score")
        ax.set_ylabel("Resume Filename")
        ax.set_title(f"Resume Ranking for '{search_query}'")
        ax.invert_yaxis()
        st.pyplot(fig)

    else:
        st.warning("⚠️ No matching resumes found.")

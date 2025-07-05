import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os

# --- NLTK punkt download fix for Streamlit Cloud ---
nltk.data.path.append('./nltk_data')  # Tell NLTK to look here
try:
    nltk.data.find('tokenizers/punkt')  # Check if 'punkt' is present
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')  # Download if not

# --- Load the dataset ---
data = pd.read_csv('product_recomendation.csv', encoding='latin1')

# Drop 'id' column if it exists
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# --- Define tokenizer and stemmer ---
stemmer = SnowballStemmer('english')

def tokenize_and_stem(text):
    try:
        tokens = nltk.word_tokenize(str(text).lower())
        stems = [stemmer.stem(t) for t in tokens if t.isalpha()]
        return stems
    except Exception as e:
        return []

# Add a column for stemmed tokens
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# TF-IDF Vectorizer with custom tokenizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)

# Cosine similarity function
def cosine_sim(text1, text2):
    text1_joined = ' '.join(text1)
    text2_joined = ' '.join(text2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_joined, text2_joined])
    return cosine_similarity(tfidf_matrix)[0][1]

# Search products function
def search_products(query):
    query_stemmed = tokenize_and_stem(query)
    similarities = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.copy()
    results['similarity'] = similarities
    results = results.sort_values(by='similarity', ascending=False).head(10)
    return results[['Title', 'Description', 'Category', 'similarity']]

# --- Streamlit app interface ---
img = Image.open('Untitled.png')
st.image(img, width=600)
st.title("Search Engine and Product Recommendation System")

query = st.text_input("Enter Product Name")
submit = st.button('Search')

if submit and query:
    results = search_products(query)
    if not results.empty:
        st.write(results)
    else:
        st.warning("No results found. Try another search.")

import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

nltk.download('punkt')  # Make sure tokenizer is available

# Load the dataset
# Option 1 - most likely to work (Windows encoding)
data = pd.read_csv('product_recomendation.csv', encoding='latin1')



# Drop 'id' column if exists
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(str(text).lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Add stemmed token column
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# TF-IDF Vectorizer with custom tokenizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)

# Define cosine similarity function
def cosine_sim(text1, text2):
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]

# Search products
def search_products(query):
    query_stemmed = tokenize_and_stem(query)
    similarities = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.copy()
    results['similarity'] = similarities
    results = results.sort_values(by='similarity', ascending=False).head(10)
    return results[['Title', 'Description', 'Category', 'similarity']]

# Streamlit app
img = Image.open('Untitled.png')
st.image(img, width=600)
st.title("Search Engine and Product Recommendation System on Product Data")

query = st.text_input("Enter Product Name")
submit = st.button('Search')

if submit and query:
    results = search_products(query)
    st.write(results)

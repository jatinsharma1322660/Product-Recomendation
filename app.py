import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# --- NLTK punkt setup for Streamlit Cloud ---
nltk.data.path.append('./nltk_data')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')

# --- Load the dataset ---
data = pd.read_csv('product_recomendation.csv', encoding='latin1')

# Ensure required columns exist
required_cols = ['Title', 'Description']
if not all(col in data.columns for col in required_cols):
    st.error("CSV must contain 'Title' and 'Description' columns.")
    st.stop()

# Drop missing rows
data = data.dropna(subset=required_cols)

# --- Define tokenizer and stemmer ---
stemmer = SnowballStemmer('english')

def tokenize_and_stem(text):
    try:
        tokens = nltk.word_tokenize(str(text).lower())
        stems = [stemmer.stem(t) for t in tokens if t.isalpha()]
        return stems
    except Exception:
        return []

# Add stemmed token column
data['stemmed_tokens'] = data.apply(
    lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']),
    axis=1
)

# TF-IDF with custom tokenizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)

# Cosine similarity function
def cosine_sim(text1, text2):
    t1 = ' '.join(text1)
    t2 = ' '.join(text2)
    if not t1.strip() or not t2.strip():
        return 0.0
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform([t1, t2])
        return cosine_similarity(tfidf_matrix)[0][1]
    except:
        return 0.0

# Search products function
def search_products(query):
    query_stemmed = tokenize_and_stem(query)
    similarities = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.copy()
    results['similarity'] = similarities

    # If no similarity > 0, fallback to top 10
    if results['similarity'].max() == 0:
        fallback = results.sort_values(by='similarity', ascending=False).head(10)
        return fallback[['Title', 'Description', 'Category', 'similarity']]
    
    filtered = results[results['similarity'] > 0]
    top_matches = filtered.sort_values(by='similarity', ascending=False).head(10)
    return top_matches[['Title', 'Description', 'Category', 'similarity']]

# --- Streamlit Interface ---
img = Image.open('Untitled.png')
st.image(img, width=600)
st.title("🔍 Product Search & Recommendation System")

query = st.text_input("Enter product name to search:")
submit = st.button("Search")

if submit and query:
    results = search_products(query)
    if not results.empty:
        st.success(f"Top {len(results)} matching products found:")
        st.dataframe(results, use_container_width=True)
    else:
        st.warning("No results found. Try a different keyword.")

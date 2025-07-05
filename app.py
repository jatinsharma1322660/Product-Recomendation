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

# Drop 'id' column if it exists
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# Drop rows with missing title or description
data = data.dropna(subset=['Title', 'Description'])

# Lowercase Category column for reliable matching
data['Category'] = data['Category'].str.lower()

# --- Define tokenizer and stemmer ---
stemmer = SnowballStemmer('english')

def tokenize_and_stem(text):
    try:
        tokens = nltk.word_tokenize(str(text).lower())
        stems = [stemmer.stem(t) for t in tokens if t.isalpha()]
        return stems
    except Exception:
        return []

# Add a column for stemmed tokens
data['stemmed_tokens'] = data.apply(
    lambda row: tokenize_and_stem(str(row.get('Title', '')) + ' ' + str(row.get('Description', ''))),
    axis=1
)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)

# Cosine similarity function
def cosine_sim(text1, text2):
    text1_joined = ' '.join(text1)
    text2_joined = ' '.join(text2)
    if not text1_joined.strip() or not text2_joined.strip():
        return 0.0
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform([text1_joined, text2_joined])
        return cosine_similarity(tfidf_matrix)[0][1]
    except ValueError:
        return 0.0

# --- Search Products ---
def search_products(query):
    query_stemmed = tokenize_and_stem(query.lower())

    # Category filter based on keyword match
    category_keywords = ['laptop', 'mobile', 'headphone', 'camera', 'speaker', 'watch', 'charger', 'audio']
    filtered_data = data.copy()

    for keyword in category_keywords:
        if keyword in query.lower():
            filtered_data = data[data['Category'].str.contains(keyword)]
            break

    # Compute similarities
    similarities = filtered_data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = filtered_data.copy()
    results['similarity'] = similarities
    results = results[results['similarity'] > 0]
    results = results.sort_values(by='similarity', ascending=False).head(10)

    columns = ['Title', 'Description', 'similarity']
    if 'Category' in results.columns:
        columns.insert(2, 'Category')

    return results[columns]

# --- Streamlit App UI ---
img = Image.open('Untitled.png')
st.image(img, width=600)
st.title("🔍 Product Recommendation System")

query = st.text_input("Enter product name or keyword (e.g. 'laptop', 'headphones')")
submit = st.button('Search')

if submit and query:
    results = search_products(query)
    if not results.empty:
        st.write(results)
    else:
        st.warning("❌ No results found. Try a different search.")

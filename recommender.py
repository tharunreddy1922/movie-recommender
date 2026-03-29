import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')

# 1. Load data
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge both datasets on title
movies = movies.merge(credits, on='title')

# Keep only useful columns
movies = movies[['movie_id','title','overview',
                 'genres','keywords','cast','crew']]
movies.dropna(inplace=True)

# 2. Parse JSON columns
def extract_names(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def extract_cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)][:3]

def extract_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['genres']   = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['cast']     = movies['cast'].apply(extract_cast)
movies['crew']     = movies['crew'].apply(extract_director)

# 3. Remove spaces inside names
def collapse(lst):
    return [i.replace(" ","") for i in lst]

movies['genres']   = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast']     = movies['cast'].apply(collapse)
movies['crew']     = movies['crew'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# 4. Create tags column
movies['tags'] = (movies['overview'] + movies['genres'] +
                  movies['keywords'] + movies['cast'] +
                  movies['crew'])
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

# 5. Stemming
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(w) for w in text.split()])

movies['tags'] = movies['tags'].apply(stem)

# 6. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()
print(f"Vector shape: {vectors.shape}")

# 7. Cosine Similarity
similarity = cosine_similarity(vectors)
print(f"Similarity matrix: {similarity.shape}")

# 8. Recommend function
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True, key=lambda x: x[1])[1:6]
    for i, _ in movie_list:
        print(movies.iloc[i]['title'])

# Test it
print("\nRecommendations for Avatar:")
recommend('Avatar')

# 9. Save model
final = movies[['movie_id','title']].reset_index(drop=True)
pickle.dump(final, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))
print("\n✅ movies.pkl and similarity.pkl saved!")
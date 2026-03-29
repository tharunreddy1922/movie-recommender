import streamlit as st
import pickle
import requests

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

movies, similarity = load_data()

def fetch_poster(movie_id):
    API_KEY = "cb46c1e3bc42a5720dee7a410dfa7e5d" # ← add your key here
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    except:
        pass
    return "https://placehold.co/500x750/1a1a2e/white?text=No+Poster"

def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]
    names, posters = [], []
    for i, _ in movie_list:
        movie_id = movies.iloc[i]['movie_id']
        names.append(movies.iloc[i]['title'])
        posters.append(fetch_poster(movie_id))
    return names, posters

# UI
st.title("🎬 Movie Recommendation System")
st.markdown("*Find movies similar to your favourites — powered by ML*")
st.divider()

selected_movie = st.selectbox(
    "Select a movie you like:",
    movies['title'].values,
    index=None,
    placeholder="e.g. Avatar, Inception, The Dark Knight..."
)

if st.button("🍿 Get Recommendations", type="primary"):
    if selected_movie:
        with st.spinner("Finding similar movies..."):
            names, posters = recommend(selected_movie)

        st.subheader(f"Because you liked **{selected_movie}**:")
        cols = st.columns(5)
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.image(poster, width=150)  # ← fixed warning
                st.caption(f"**{name}**")
    else:
        st.warning("Please select a movie first!")

st.divider()
st.markdown(
    "<div style='text-align:center;color:gray'>Built with Streamlit · Data from TMDB · ML with scikit-learn</div>",
    unsafe_allow_html=True
)
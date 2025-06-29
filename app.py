import streamlit as st
import pickle
import pandas as pd
import requests
import time

# ✅ TMDB API key (keep private for deployment)
api_key = 'ad3cceb1f043758f433fc7239c437141'


# ✅ Fetch movie poster from TMDB
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    except requests.exceptions.RequestException as e:
        print(f"Poster fetch failed for movie_id {movie_id}: {e}")
    return "https://via.placeholder.com/500x750?text=No+Image"


# ✅ Load data
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movie = pd.DataFrame(movie_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))


# ✅ Recommend movies based on similarity
def recommend(selected_movie):
    try:
        movie_index = movie[movie['title'] == selected_movie].index[0]
    except IndexError:
        st.error("Selected movie not found.")
        return [], []

    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        index = i[0]
        movie_row = movie.iloc[index]

        # Ensure ID is valid
        if 'id' not in movie_row or pd.isna(movie_row['id']):
            continue

        movie_id = int(movie_row['id'])
        poster_url = fetch_poster(movie_id)

        # Skip if fallback placeholder is returned
        if "placeholder.com" in poster_url:
            continue

        recommended_movies.append(movie_row['title'])
        recommended_posters.append(poster_url)

        if len(recommended_movies) == 5:
            break

    return recommended_movies, recommended_posters



# ✅ Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title('🎬 Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select a movie to get recommendations:',
    movie['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    if len(names) < 5:
        st.warning("Some recommendations could not be fetched due to missing data.")

    cols = st.columns(5)
    for i in range(len(names)):
        with cols[i]:
            st.image(posters[i])
            st.caption(names[i])


import os
import gdown
import pickle
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

session = requests.Session()


# @st.cache_data
def load_movies_df(movie_df_path, poster_df_path) -> pd.DataFrame():
    """
    Добавляет постеры к фильмам к датафрейму с фильмами
    """

    with (open(movie_df_path, "rb")) as f:
        movie_df = pickle.load(f)

    # Load poster dataframe from csv
    posters_df = pd.read_csv(poster_df_path, sep="\t")

    # Prepare poster dataframe
    # Fill NaN values with empty string
    posters_df.fillna("", inplace=True)
    # Remove year from the title (Fight Club (1999) -> Fight Club)
    posters_df["title"] = posters_df["title"].apply(lambda x: x[:-7])
    # Remove "The" from the end of the title (Shaw shank Redemption, The -> Shaw shank Redemption)
    posters_df["title"] = posters_df["title"].apply(lambda x: x[:-5] if x.endswith(", The") else x)

    # Merge two dataframes by title
    movies_df = movie_df.merge(posters_df, on="title", how="left")

    # Remove rows with no poster (None values)
    movies_df = movies_df[movies_df["poster_link"].notna()]
    # Drop year_y column
    movies_df.drop(columns=["year_y"], inplace=True)
    # Rename year_x to year
    movies_df.rename(columns={"year_x": "year"}, inplace=True)
    # Rename id to tmdb_id
    movies_df.rename(columns={"id": "tmdb_id"}, inplace=True)
    # Drop item column
    movies_df.drop(columns=["item"], inplace=True)
    # Reset index
    movies_df.reset_index(drop=True, inplace=True)

    return movies_df


# @st.cache_data
def load_users_df(movies_df: pd.DataFrame, ratings_path: str, links_path: str):
    """
    Добавляет оценки пользователей к датафрейму с фильмами
    """

    # Скачивание ratings.csv из Google Drive если его нет в папке data/datasets
    if not os.path.exists(ratings_path):
        url = "https://drive.google.com/uc?id=1HlvAy6BnH7IQfzceLXSOtJ9Ns9tOG3_r"
        gdown.download(url=url, output=ratings_path, fuzzy=True, resume=True, quiet=False)
        rating_df = pd.read_csv(ratings_path)
    else:
        rating_df = pd.read_csv(ratings_path)

    links_df = pd.read_csv(links_path)

    # Переименовываем столбец с id фильма в links_df
    links_df.rename(columns={"tmdbId": "tmdb_id"}, inplace=True)

    # Объединяем датасеты по id фильма
    tmp_df = movies_df.merge(links_df, on='tmdb_id')

    # Удаляем дубликаты и пропущенные значения
    tmp_df = tmp_df.drop_duplicates(subset="tmdb_id")
    tmp_df = tmp_df.dropna(subset="movieId")

    # Объединяем датасеты по id фильма
    users_df = rating_df.merge(tmp_df, on='movieId')

    return users_df


# @st.cache_data()
def compute_similarity_matrix(df):
    """
    Compute similarity matrix.
    
    Parameters
    ----------
    df : pandas.DataFrame
            Movies_df dataframe.
    
    Returns
    -------
    similarity : numpy.ndarray
        Similarity matrix.    
    """

    cv = CountVectorizer(max_features=5000, stop_words="english")

    vector = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vector)

    return similarity


def add_movie(movie):
    """
    Add movie to the list of selected movies.
    
    Parameters
    ----------
    movie : dict
        Movie dictionary.
            
    Returns
    -------
    None
    """
    if st.session_state["clicked"]:
        st.session_state["selected_movie_count"] += 1
        st.session_state["added_movie_ids"].append(movie["tmdb_id"])
        st.session_state["clicked"] = False


def set_status(status):
    """
    Set status.
    
    Parameters
    ----------
    status : bool
        Status.
    
    Returns
    -------
    None
    """
    st.session_state["status"] = status


def capture_return(_):
    """
    Capture return key (to prevent unexpected events by frequent clicking)
    
    Parameters
    ----------
    _ : event        .
        
    Returns
    -------
    None
    """
    st.session_state["clicked"] = True


def retry():
    """
    Reset session state.
    
    Parameters
    ----------
    
    Returns
    -------
    None
    """
    st.session_state["selected_movie_count"] = 0
    st.session_state["added_movie_ids"] = []
    st.session_state["status"] = False


def set_value(key):
    """
    Set value.
    
    Parameters
    ----------
    key : str
        Key.
    
    Returns
    -------
    None
    """
    st.session_state[key] = st.session_state["key_" + key]


def fetch_poster(movie_id):
    """
    Fetch poster image from TMDB API.

    Parameters
    ----------
    movie_id : int
        Movie ID.
    
    Returns
    -------
    full_path : str
        Full path to the poster image.
    """

    try:
        url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
            movie_id)

        data = session.get(url)

        data.raise_for_status()

        data = data.json()
        poster_path = data['poster_path']
        if poster_path is None:
            return None
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

        return full_path

    except requests.exceptions.HTTPError as err:
        print(err)
        return None

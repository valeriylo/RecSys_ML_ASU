import pickle
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

session = requests.Session()


def load_data(movie_df_path):
    """
    Load data from pickle file.
    
    Parameters
    ----------
    movie_df_path : str
        Path to the pickle file.
    
    Returns
    -------
    df : pandas.DataFrame
    """

    # Load main dataframe from pickle
    df = pd.read_csv(movie_df_path)

    return df


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
        st.session_state["added_movie_ids"].append(movie["id"])
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


def compute_ranking():
    """


    Returns
    -------

    """

import pandas as pd
import requests
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

session = requests.Session()


def load_data(movie_df_path, poster_df_path) -> pd.DataFrame():
    """
    Load data from pickle and csv files.
    
    Parameters
    ----------
    movie_df_path : str
        Path to the pickle file.
    poster_df_path : str
        Path to the csv file.
    
    Returns
    -------
    df : pandas.DataFrame
        Merged dataframe.
    """

    # Load main dataframe from pickle
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
    df = movie_df.merge(posters_df, on="title", how="left")

    st.write(df)

    # Remove rows with no poster (None values)
    df = df[df["poster_link"].notna()]
    # Drop year_y column
    df.drop(columns=["year_y"], inplace=True)
    # Rename year_x to year
    df.rename(columns={"year_x": "year"}, inplace=True)
    # Drop item column
    df.drop(columns=["item"], inplace=True)
    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Count number of lost movies from the original dataframe movie_df
    lost_movies = len(movie_df) - len(df)
    # print("Lost movies: ", lost_movies)

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

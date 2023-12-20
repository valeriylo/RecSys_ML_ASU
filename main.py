import streamlit as st
from PIL import Image
from src.utils import add_movie, set_status, capture_return, retry, set_value, load_movies_df, load_users_df
from src.predict import get_random_rec, get_content_rec, get_content_rank_rec

st.set_page_config(page_title="Movie Recommender", layout="wide")

STATE_KEYS_VALS = [
    ("selected_movie_count", 0),  # main view
    ("added_movie_ids", []),  # main view
    ("status", False),
    ("clicked", False),
    ("input_len", 10),  # sidebar view
    ("top_k", 10),  # sidebar view
    ("years", (1990, 2010)),  # sidebar view
    ("model_type", "Content-based"),  # sidebar view
]
for k, v in STATE_KEYS_VALS:
    if k not in st.session_state:
        st.session_state[k] = v

# Define functions of sidebar
st.sidebar.title("Setting")
years = st.sidebar.slider(
    "Select a range of years",
    min_value=1900,
    max_value=2015,
    value=st.session_state["years"],
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("years",),
    key="key_years",
)

val = st.sidebar.number_input(
    "How many movies do you want to choose?",
    format="%i",
    min_value=5,
    max_value=20,
    value=int(st.session_state["input_len"]),
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("input_len",),
    key="key_input_len",
)

st.sidebar.number_input(
    "How many movies do you want to get recommended?",
    format="%i",
    min_value=5,
    max_value=20,
    value=int(st.session_state["top_k"]),
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("top_k",),
    key="key_top_k",
)

# Select model type in sidebar (make selectbox uneditable on user click )
model_type = st.sidebar.selectbox(
    "Select a model type",
    ("Random", "Content-based", "Content-based with ranking"),
    index=1,
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("model_type",),
    key="key_model_type",
)

st.sidebar.button(
    "START",
    on_click=set_status,
    args=(True,),
    disabled=st.session_state["status"],
)

# Load movies only data
movies_df = load_movies_df("./data/datasets/movies_df.pickle", "./data/datasets/poster.csv")

# Load users data with movies and ratings (warning: this is a large file, may affect performance)
# users_df = load_users_df(movies_df, "./data/datasets/ratings.csv", "./data/datasets/links.csv")

# Define functions of main part

st.title("Movies Recommender with Streamlit")
no_image = Image.open("data/placeholder.png")
# When the start button has been clicked from the sidebar
if st.session_state["status"]:
    unique_key = 0
    movies_df = movies_df[~movies_df["title"].isin(st.session_state["added_movie_ids"])]
    movies_df = movies_df[(movies_df["year"] >= years[0]) & (movies_df["year"] <= years[1])]
    # Remove duplicates and reset index
    movies_df.drop_duplicates(subset=["title"], inplace=True)
    movies_df.reset_index(drop=True, inplace=True)

    random_movies = movies_df.sample(20)

    selection_container = st.empty()

    # While the number of chosen movies is less than the target number
    if st.session_state["selected_movie_count"] < st.session_state["input_len"]:
        with selection_container.container():
            st.subheader(
                "Please select the favorite movie({}-{}): {}/{}".format(
                    st.session_state["years"][0],
                    st.session_state["years"][1],
                    st.session_state["selected_movie_count"],
                    st.session_state["input_len"],
                )
            )

            for row_index in range(2):
                for col_index, col in enumerate(st.columns(10)):
                    movie = random_movies.iloc[unique_key]
                    title = movie["title"]
                    movie_id = movies_df[movies_df['title'] == title]['tmdb_id'].values[0]
                    poster = movie["poster_link"] if movie["poster_link"] else no_image
                    unique_key += 1

                    with col:
                        st.image(poster)
                        capture_return(
                            st.checkbox(
                                title,
                                key="mov-{}".format(movie["tmdb_id"]),
                                on_change=add_movie,
                                args=(movie,),
                            )
                        )
    # When the required number of movies is chosen
    else:
        # Empty the above view
        selection_container.empty()

        with st.container():
            st.subheader("Random Movies")
            with st.spinner("Wait for it..."):
                random_rec_movies = get_random_rec(movies_df, st.session_state["top_k"])
            # Render recommended movies
            for col_index, col in enumerate(st.columns(st.session_state["top_k"])):
                movie = random_rec_movies.iloc[col_index]
                title = movie["title"]
                movie_id = movies_df[movies_df['title'] == title]['tmdb_id'].values[0]
                poster = movie["poster_link"] if movie["poster_link"] else no_image

                with col:
                    st.image(poster)
                    st.caption(title)

        with st.container():
            st.subheader("Recommended Movies")
            with st.spinner("Wait for it..."):
                s3_rec_movies = get_content_rec(
                    movies_df,
                    st.session_state["added_movie_ids"],
                    st.session_state["top_k"],
                )

            # Render recommended movies
            for col_index, col in enumerate(st.columns(st.session_state["top_k"])):
                movie = s3_rec_movies.iloc[col_index]
                title = movie["title"]
                movie_id = movies_df[movies_df['title'] == title]['tmdb_id'].values[0]
                poster = movie["poster_link"] if movie["poster_link"] else no_image
                with col:
                    st.image(poster)
                    st.caption(title)

        (_, center, _) = st.columns([4, 1, 4])
        with center:
            st.button(
                "Retry",
                on_click=retry,
            )


else:
    st.header("&#9664; Waiting for the setting....")

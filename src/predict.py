import pandas as pd
import streamlit as st
from src.utils import compute_similarity_matrix


def get_random_rec(df, top_k):
    """
    Get random recommendations.
    
    Parameters
    ----------
    df : pandas.DataFrame
            Movie Dataframe.
    top_k : int
            Number of recommendations.
    
    Returns
    -------
    recs : pandas.DataFrame
            Recommended movies.
    """
    top_k = int(top_k)
    recs = df.sample(top_k)
    return recs


def get_content_rec(df, input_ids, top_k):
    """
    Get basic collaborative filtering recommendations.
    
    Parameters
    ----------
    df : pandas.DataFrame
            Movie Dataframe.
    input_ids : list
            Indecis of input movie ids.
    top_k : int
            Number of recommendations.
    
    Returns
    -------
    recs : pandas.DataFrame
            Recommended movies.
    """

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(df)
    
    # Store the movie indices and scores
    movie_indices = []
    scores = []
    # Iterate over the input movie ids
    for input_id in input_ids:
        # Get the index of the input movie
        idx = df[df["tmdb_id"] == input_id].index[0]
        # Get the similarity scores of the input movie
        sim_scores = list(enumerate(similarity_matrix[idx]))
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the movie indices

        movie_indices.extend([i[0] for i in sim_scores])
        # Get the movie scores
        scores.extend([i[1] for i in sim_scores])
    
    # Create dataframe of movie indices and scores
    df_scores = pd.DataFrame({"idx": movie_indices, "score": scores})
    # Sort the dataframe based on the scores
    df_scores.sort_values(by="score", ascending=False, inplace=True)
    # Get the top k movie indices
    top_k_indices = df_scores["idx"].values[:top_k]
    # Get the top k movies
    recs = df.iloc[top_k_indices]

    return recs


def get_content_rank_rec(df, input_ids, top_k):
    """
    Get advanced collaborative filtering recommendations.
    Use the ratings data to get the top k movies. Also, use the 
    
    Parameters
    ----------
    df : pandas.DataFrame
            Dataframe.
    input_ids : list
            List of input movie ids.
    top_k : int
            Number of recommendations.
    
    Returns
    -------
    recs : pandas.DataFrame
            Recommended movies.
    """

    # If popularity ratio is hight and raiting is high for cos score, assign highter position in ranking

    return get_content_rec(df, input_ids, top_k)




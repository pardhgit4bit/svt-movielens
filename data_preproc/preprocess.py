"""
preprocess.py
Step 1: load MovieLens 100K and build the user-movie ratings matrix M
and the observation mask (Omega).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-100k"
N_USERS = 943
N_MOVIES = 1682


def load_ratings(path=DATA_DIR / "u.data"):
    """Read u.data and return 0-indexed (users, movies, ratings) arrays."""
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user", "movie", "rating", "timestamp"],
    )
    users = df["user"].to_numpy() - 1      # IDs start at 1, indices start at 0
    movies = df["movie"].to_numpy() - 1
    ratings = df["rating"].to_numpy(dtype=float)
    return users, movies, ratings


def build_matrix(users, movies, ratings, shape=(N_USERS, N_MOVIES)):
    """
    Build the ratings matrix M and boolean mask of observed entries.
    M[i, j] = rating of user i for movie j (0 where unobserved).
    mask[i, j] = True exactly when (i, j) is in Omega.
    """
    M = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    M[users, movies] = ratings
    mask[users, movies] = True
    return M, mask


if __name__ == "__main__":
    users, movies, ratings = load_ratings()
    M, mask = build_matrix(users, movies, ratings)

    # Sanity checks (the script stops with an error if any fails)
    assert M.shape == (N_USERS, N_MOVIES)
    assert mask.sum() == len(ratings) == 100_000, "duplicate or missing ratings"
    assert ratings.min() == 1 and ratings.max() == 5
    assert users.min() == 0 and users.max() == N_USERS - 1
    assert movies.min() == 0 and movies.max() == N_MOVIES - 1
    assert len(np.unique(users)) == N_USERS
    assert len(np.unique(movies)) == N_MOVIES

    print("Matrix shape        :", M.shape)
    print("Observed ratings    :", mask.sum())
    print("Observed fraction   : {:.2%}".format(mask.mean()))
    print("Mean rating         : {:.3f}".format(ratings.mean()))
    print("Ratings per user    : min {}, max {}".format(
        mask.sum(axis=1).min(), mask.sum(axis=1).max()))
    print("Ratings per movie   : min {}, max {}".format(
        mask.sum(axis=0).min(), mask.sum(axis=0).max()))
    print("User index range    : {} to {}".format(users.min(), users.max()))
    print("Movie index range   : {} to {}".format(movies.min(), movies.max()))
    print("Unique users        :", len(np.unique(users)))
    print("Unique movies       :", len(np.unique(movies)))
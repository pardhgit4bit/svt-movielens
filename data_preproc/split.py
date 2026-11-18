"""
split.py
Step 2: split the observed ratings into a final test set and a tuning pool,
then cut the pool into K cross-validation folds.
"""
import numpy as np

from data_preproc.preprocess import load_ratings, build_matrix, N_USERS, N_MOVIES

SEED = 42
TEST_FRACTION = 0.20
N_FOLDS = 5


def split_indices(n_ratings, test_fraction=TEST_FRACTION,
                  n_folds=N_FOLDS, seed=SEED):
    """
    Randomly split rating positions 0..n_ratings-1 into
    a final test set and n_folds disjoint folds of the remaining pool.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ratings)
    n_test = int(round(test_fraction * n_ratings))
    test_idx = perm[:n_test]
    pool_idx = perm[n_test:]
    folds = np.array_split(pool_idx, n_folds)
    return test_idx, pool_idx, folds


def fold_train_val(folds, k):
    """Validation = fold k, training = all the other folds."""
    val_idx = folds[k]
    train_idx = np.concatenate([f for i, f in enumerate(folds) if i != k])
    return train_idx, val_idx


def count_cold_start(train_idx, eval_idx, users, movies):
    """Count eval ratings whose user / movie has no training rating."""
    seen_users = np.zeros(N_USERS, dtype=bool)
    seen_movies = np.zeros(N_MOVIES, dtype=bool)
    seen_users[users[train_idx]] = True
    seen_movies[movies[train_idx]] = True
    cold_users = (~seen_users[users[eval_idx]]).sum()
    cold_movies = (~seen_movies[movies[eval_idx]]).sum()
    return cold_users, cold_movies


if __name__ == "__main__":
    users, movies, ratings = load_ratings()
    n = len(ratings)
    test_idx, pool_idx, folds = split_indices(n)

    # Sanity checks
    assert len(test_idx) + len(pool_idx) == n
    assert len(np.intersect1d(test_idx, pool_idx)) == 0
    assert sorted(np.concatenate(folds)) == sorted(pool_idx)
    for a in range(N_FOLDS):
        for b in range(a + 1, N_FOLDS):
            assert len(np.intersect1d(folds[a], folds[b])) == 0

    print("Total ratings       :", n)
    print("Final test set      :", len(test_idx))
    print("Tuning pool         :", len(pool_idx))
    print("Fold sizes          :", [len(f) for f in folds])

    cu, cm = count_cold_start(pool_idx, test_idx, users, movies)
    print("\nFinal test (train = pool):")
    print("  cold-start users  :", cu)
    print("  cold-start movies :", cm)

    print("\nCross-validation:")
    for k in range(N_FOLDS):
        train_idx, val_idx = fold_train_val(folds, k)
        M_train, mask_train = build_matrix(
            users[train_idx], movies[train_idx], ratings[train_idx])
        cu, cm = count_cold_start(train_idx, val_idx, users, movies)
        print("  fold {}: train {}, val {}, observed in M_train {}, "
              "cold users {}, cold movies {}".format(
                  k + 1, len(train_idx), len(val_idx),
                  mask_train.sum(), cu, cm))
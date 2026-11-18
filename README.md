# SVT for Movie Recommendation (MovieLens 100K)

Singular Value Thresholding (SVT) for matrix completion, applied to the
MovieLens 100K ratings dataset.

## Repository structure

    data/                  MovieLens files (not in the repo, see Dataset)
    data_preproc/
        preprocess.py      builds the ratings matrix and observation mask
        split.py           final test set and cross-validation folds
    requirements.txt

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1      # Windows PowerShell
    pip install -r requirements.txt

On macOS/Linux, activate with `source .venv/bin/activate` instead.
All commands below are run from the repository root.

## Dataset

This project uses the **MovieLens 100K** dataset (943 users, 1682 movies,
100,000 ratings) from GroupLens Research.

Download it from: https://files.grouplens.org/datasets/movielens/ml-100k.zip

Unzip it so the files are at:

    data/ml-100k/u.data

The dataset is **not included in this repository**, since the MovieLens
license does not permit redistribution without permission. The `data/`
folder is listed in `.gitignore`.

**Citation:** F. M. Harper and J. A. Konstan. *The MovieLens Datasets: History
and Context.* ACM Transactions on Interactive Intelligent Systems, 5(4),
Article 19, 2015. https://doi.org/10.1145/2827872

## Data Preprocessing

`data_preproc/preprocess.py` reads `data/ml-100k/u.data`, which contains one
rating per line as `user_id, movie_id, rating, timestamp`.

It then:

1. Converts the 1-based user and movie IDs to 0-based indices.
2. Builds the ratings matrix `M` of size 943 × 1682, where `M[i, j]` is the
   rating of user `i` for movie `j`, and 0 if unobserved.
3. Builds a boolean mask marking the observed entries, i.e. the set Ω.

Only 6.3% of the entries are observed. These are the entries SVT will use
to complete the matrix.

Run it with:

    python -m data_preproc.preprocess

### Dataset statistics

    Matrix shape        : (943, 1682)
    Observed ratings    : 100000
    Observed fraction   : 6.30%
    Mean rating         : 3.530
    Ratings per user    : min 20, max 737
    Ratings per movie   : min 1, max 583
    User index range    : 0 to 942
    Movie index range   : 0 to 1681
    Unique users        : 943
    Unique movies       : 1682

Every user has rated at least 20 movies, but some movies have only a
single rating. This matters when splitting into training and test sets.

### Design decisions

**Why is the matrix not saved as a CSV (or pushed to GitHub)?**

- *License.* A CSV of the matrix contains every rating in the dataset, just
  rearranged. It is the same data as `u.data`, so publishing it is still
  redistribution, which the MovieLens license does not permit without
  permission. The repository therefore contains only code, and everyone
  downloads the data from GroupLens themselves.
- *It is wasteful.* About 94% of the entries are zeros, so a dense CSV would be
  a large file that is almost entirely empty.
- *It is unnecessary.* Building the matrix from `u.data` takes a fraction of
  a second.
- *It is the wrong object for the later steps.* Cross-validation needs a
  different training matrix for every fold, so one saved full matrix would
  only be useful for this sanity check. `build_matrix` is a function so that
  it can be reused on any subset of ratings.

**Are the user and movie indices correct?**

In MovieLens 100K, user IDs run from 1 to 943 and movie IDs from 1 to 1682,
with no gaps. Subtracting 1 therefore gives contiguous indices
0..942 and 0..1681, so:

- row `i` of `M` is the user with ID `i + 1`;
- column `j` of `M` is the movie with ID `j + 1`, whose title is on line
  `j + 1` of `data/ml-100k/u.item`.

This is checked by assertions in `preprocess.py`, and the verified values
(index ranges and the number of unique users and movies) are printed in the
statistics above. The check matters because larger MovieLens versions have
gaps in the movie IDs, and there the IDs would need to be remapped to
contiguous indices instead of just shifted by 1.

**Why is 0 used for unobserved entries?**

Ratings are 1 to 5, so 0 cannot be confused with a real rating. The boolean
`mask` is what defines Ω, and SVT uses the mask, not the zeros, to decide
which entries are known.

## Train / Test Split and Cross-Validation

SVT has parameters (the threshold τ, the step size δ, and others) that must
be chosen without looking at the final test data. `data_preproc/split.py`
therefore creates:

1. a **final test set** of 20% of the ratings (20,000), used once at the
   very end;
2. a **tuning pool** of the remaining 80% (80,000), cut into **5 disjoint
   folds** of 16,000 ratings. For each fold, SVT is trained on the other
   four folds (64,000 ratings) and scored on that fold.

The split is random with a fixed seed (42), so the results are reproducible.

Run it with:

    python -m data_preproc.split

### Split statistics

    Total ratings       : 100000
    Final test set      : 20000
    Tuning pool         : 80000
    Fold sizes          : [16000, 16000, 16000, 16000, 16000]

    Final test (train = pool):
      cold-start users  : 0
      cold-start movies : 36

    Cross-validation:
      fold 1: train 64000, val 16000, observed in M_train 64000, cold users 0, cold movies 39
      fold 2: train 64000, val 16000, observed in M_train 64000, cold users 0, cold movies 18
      fold 3: train 64000, val 16000, observed in M_train 64000, cold users 0, cold movies 39
      fold 4: train 64000, val 16000, observed in M_train 64000, cold users 0, cold movies 39
      fold 5: train 64000, val 16000, observed in M_train 64000, cold users 0, cold movies 32

### Design decisions

**Why split individual ratings, not users or movies?**
Matrix completion predicts missing entries for users and movies that are
already in the matrix, so the held-out ratings are random entries of `M`.

**Why keep a final test set in addition to cross-validation?**
The cross-validation score is used to pick parameters, so it becomes
optimistic about the chosen parameters. The final test set is never used for
tuning, so it gives an honest estimate.

**What are cold-start entries?**
A held-out rating is cold-start if its user or movie has no rating in the
training matrix. SVT cannot learn such a row or column, since it has no
observed entries. There are no cold-start users (every user has at least
20 ratings), but a few cold-start movies: 36 of 20,000 in the final test
set (0.18%), and 18 to 39 of 16,000 in the validation folds. They are
counted and reported. How they are predicted is decided when the SVT
evaluation is written.

**How is the training matrix built?**
`build_matrix` from `preprocess.py` is called on the training ratings only,
so held-out ratings never enter the matrix SVT sees (the `observed in
M_train` column above confirms 64,000 entries per fold).
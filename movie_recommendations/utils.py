import numpy as np
import pandas as pd


def get_movie_names(movies, movie_ids):
    return movies.set_index("movie_id").movie[movie_ids]


def get_similar_movies(movies, movie_id, num_movies):
    movie_content = np.array(movies.iloc[:, 4:])
    # get corresponding row number of movie
    try:
        movie_idx = np.where(movies.movie_id == movie_id)[0][0]
    except IndexError:
        print("Movie is not in the database")
        return None

    # calculate similarity
    dot_prod_movies = np.dot(movie_content[movie_idx], np.transpose(movie_content))

    # find most similar movies
    similar_idxs = np.where(dot_prod_movies == max(dot_prod_movies))[0]

    # get corresponding movie titles
    movie_names = movies.iloc[similar_idxs, :].set_index(["movie_id"])["movie"]

    return movie_names[:num_movies]


def popular_recommendations(movies, reviews, num_recs=5):
    """TO DO"""
    reviews_copy = reviews.copy()
    movies_copy = movies.copy()

    rating_aggs_df = reviews_copy.groupby("movie_id").agg(
        {"rating": ["mean", "count"], "date": "max"}
    )
    rating_aggs_df.columns = rating_aggs_df.columns.get_level_values(
        1
    )  # collapse mult-index
    rating_aggs_sorted_df = rating_aggs_df.sort_values(
        ["mean", "count", "max"], ascending=[False, False, False]
    )
    rating_aggs_n_top = rating_aggs_sorted_df.query("count >= 5")[:num_recs]

    movies_copy.set_index(keys=["movie_id"], inplace=True)
    top_movies = movies_copy.movie[rating_aggs_n_top.index]
    movies_copy.reset_index(inplace=True)

    return top_movies


def predict_rating(user_matrix, movie_matrix, user_id, movie_id, measures_df):
    """TO DO"""

    user_id_vals = measures_df.loc[:, "user_id"].unique()
    movie_id_vals = measures_df.loc[:, "movie_id"].unique()

    try:
        user_idx = np.where(user_id_vals == user_id)[0][0]
        movie_idx = np.where(movie_id_vals == movie_id)[0][0]
        return np.dot(user_matrix[user_idx, :], movie_matrix[:, movie_idx])
    except IndexError:
        print("Either the user or movie is missing from the our database")
        return None


def get_user_item_mat(df):
    """TO DO"""
    num_users = df.iloc[:, 0].nunique()
    num_items = df.iloc[:, 1].nunique()
    # user_item_mat = np.empty((num_users, num_items))
    # user_item_mat.fill(np.nan)

    user_item_mat = np.full((num_users, num_items), fill_value=np.nan)

    user_id_lookup = dict(zip(df.iloc[:, 0].unique(), range(num_users)))
    item_id_lookup = dict(zip(df.iloc[:, 1].unique(), range(num_items)))

    users_keys = user_id_lookup.keys()
    items_keys = item_id_lookup.keys()

    for idx, row in df.iterrows():
        user_item_mat[user_id_lookup[row[0]], item_id_lookup[row[1]]] = row[2]
    return user_item_mat, users_keys, items_keys


def create_user_item_matrix(reviews):
    """TO DO"""
    user_items = reviews[["user_id", "movie_id", "rating"]]
    user_item_mat, users_keys, items_keys = get_user_item_mat(user_items)

    # return user_item_mat, users_keys, items_keys
    return user_item_mat


def funkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=100):
    """TO DO"""

    num_users = ratings_mat.shape[0]
    num_movies = ratings_mat.shape[1]
    num_ratings = np.count_nonzero(~np.isnan(ratings_mat))

    users_latent = np.random.random_sample((num_users, latent_features))
    movies_latent = np.random.random_sample((latent_features, num_movies))

    for iter_num in range(iters):
        cum_sse = 0
        for row in range(num_users):
            for col in range(num_movies):
                if ratings_mat[row, col] > 0:  # rating exists
                    predicted_value = np.dot(
                        users_latent[row, :], movies_latent[:, col]
                    )
                    error = ratings_mat[row, col] - predicted_value
                    cum_sse += np.power(error, 2)

                    weight = learning_rate * 2 * (error)
                    users_latent[row, :] += weight * movies_latent[:, col]
                    movies_latent[:, col] += weight * users_latent[row, :]
        print(f"Iteration: {iter_num + 1}   MSE: {cum_sse/num_ratings:.6f}")

    return users_latent, movies_latent


def get_blended_recommendations(
    train_df,
    user_mat,
    movie_mat,
    movies,
    find_id,
    find_id_type,
    num_recs,
):
    """TO DO"""

    train_df_user_id = train_df.user_id.unique()
    train_df_user_id_lookup = dict(zip(train_df_user_id, range(len(train_df_user_id))))

    train_df_movie_id = train_df.movie_id.unique()
    train_df_id_movie_lookup = dict(
        zip(range(len(train_df_movie_id)), train_df_movie_id)
    )

    if find_id_type == "user":
        if find_id in train_df_user_id:
            user_idx = train_df_user_id_lookup[find_id]
            predicted_ratings_user_id = np.dot(user_mat[user_idx, :], movie_mat)
            ranked_movie_idx = predicted_ratings_user_id.argsort()[-num_recs:][::-1]
            ranked_movie_ids = [
                train_df_id_movie_lookup[movie_idx] for movie_idx in ranked_movie_idx
            ]
            ranked_movie_names = get_movie_names(movies, ranked_movie_ids)
        else:
            ranked_movie_names = popular_recommendations(movies, train_df, num_recs)
    else:
        ranked_movie_names = get_similar_movies(movies, find_id, num_recs)

    return ranked_movie_names

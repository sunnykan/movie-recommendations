import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
from utils import create_user_item_matrix, funkSVD
from utils import popular_recommendations, predict_rating, get_blended_recommendations


class Recommender:
    def __init__(self, reviews_path, movies_path):
        self.reviews = pd.read_csv(reviews_path, index_col=["Unnamed: 0"])
        self.movies = pd.read_csv(movies_path, index_col=["Unnamed: 0"])

    def fit(self, latent_features=10, learning_rate=0.0001, iters=100):
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        ratings_mat = create_user_item_matrix(self.reviews)
        self.user_mat, self.movie_mat = funkSVD(
            ratings_mat, latent_features, learning_rate, iters
        )

    def predict_movie_rating(self, user_id, movie_id):
        prediction = predict_rating(
            self.user_mat, self.movie_mat, user_id, movie_id, self.reviews
        )
        return prediction

    def make_recommendations(self, find_id, find_id_type="user", num_recs=5):
        ranked_movie_names = get_blended_recommendations(
            self.reviews,
            self.user_mat,
            self.movie_mat,
            self.movies,
            find_id,
            find_id_type=find_id_type,
            num_recs=num_recs,
        )
        return ranked_movie_names


if __name__ == "__main__":
    reviews_path = "../data/train.csv"
    movies_path = "../data/movies_clean.csv"

    rec = Recommender(reviews_path, movies_path)
    rec.fit(latent_features=10, learning_rate=0.0001, iters=10)

    prediction = rec.predict_movie_rating(user_id=8, movie_id=2844)
    print(f"Prediction (user_id=8, movie_id=2844): {prediction}")

    prediction = rec.predict_movie_rating(user_id=5, movie_id=2844)
    print(f"Prediction (user_id=5, movie_id=2844): {prediction}")

    print(rec.make_recommendations(8, "user"))  # user in the dataset
    print(rec.make_recommendations(1, "user"))  # user not in dataset
    print(rec.make_recommendations(1853728, "movie"))  # movie in the dataset
    print(rec.make_recommendations(1, "movie"))  # movie not in dataset

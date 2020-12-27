import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, nargs=1)
parser.add_argument('--test', required=True, nargs=1)
args = parser.parse_args()

train_file = args.train[0]
test_file = args.test[0]
prediction_file = "complete_prediction.txt"


"""
train_file = "D:/Collaborative_Filtering/data/TrainingRatings.txt"
test_file =  "D:/Collaborative_Filtering/data/temp_test.txt"
prediction_file = "complete_predictions.txt"
test_data = pd.read_csv(test_file,sep=",",header=None)
test_data.columns = ["movie_id", "user", "original"]
test_data.sort_values(['movie_id', 'user'], ascending=[True, False])
test_data.to_csv("TestingRatings.csv")
"""

user_to_movie_mapping = dict()
movie_to_user_mapping = dict()

def training(train_file):
    with open(train_file, "r") as file:
        for each_file in file:
            movie, user, rating = each_file.split(",")
            movie = int(float(movie))
            user = int(float(user))
            rating = np.float(rating.strip())
            try:
                user_to_movie_mapping[user][movie] = rating
            except KeyError:
                user_to_movie_mapping[user] = {movie: rating}
            try:
                movie_to_user_mapping[movie][user] = rating
            except KeyError:
                movie_to_user_mapping[movie] = {user: rating}
    return user_to_movie_mapping,movie_to_user_mapping


def get_all_user_averages(dict_values):
    sum_list = list()
    keys = dict_values.values()
    for i in keys:
        sum_list.append(i.values())
    sum_list = [item for sublist in sum_list for item in sublist]
    return np.sum(list(sum_list)) / np.float(len(sum_list))

def get_user_rating(user, intersection_list, movie_to_user_mapping):
        user_ratings = []
        for i in intersection_list:
            user_ratings.append(movie_to_user_mapping[i][user])
        return user_ratings

def get_mean(user, intersection_list, movie_to_user_mapping):
        ratings = get_user_rating(user, intersection_list, movie_to_user_mapping)
        return (np.sum(list(ratings))) / (float(len(ratings)))


def pearson_cofficient_get_weights(user_test, user_watched_movie, user_to_movie_mapping, movie_to_user_mapping):
    intersection_list = np.intersect1d(user_to_movie_mapping[user_test].keys(), user_to_movie_mapping[user_watched_movie].keys())
    v_i = get_user_rating(user_test, intersection_list, movie_to_user_mapping)
    v_bar_i = get_mean(user_test, intersection_list, movie_to_user_mapping)
    v_j = get_user_rating(user_watched_movie, intersection_list, movie_to_user_mapping)
    v_bar_j = get_mean(user_watched_movie, intersection_list, movie_to_user_mapping)
    numerator = np.sum((v_i - v_bar_i) * (v_j - v_bar_j))
    denominator = np.sqrt(np.sum((v_i - v_bar_i) ** 2) * np.sum((v_j - v_bar_j) ** 2))
    if numerator <= 0 or denominator <= 0:
        return 0
    else:
        return numerator/float(denominator)


def predictions(user_test, movie_test, user_to_movie_mapping, movie_to_user_mapping):
    num = []
    w_ij = []
    list_user_rating = movie_to_user_mapping[movie_test]
    for user_watched_movie in list_user_rating:
        v_ij = np.array(list_user_rating[user_watched_movie])
        user_avg_movie_ratings = user_to_movie_mapping[user_watched_movie].values()
        vi_bar = (np.sum(list(user_avg_movie_ratings))) / np.float(len(user_avg_movie_ratings))
        weights_ij = pearson_cofficient_get_weights(user_test, user_watched_movie, user_to_movie_mapping, movie_to_user_mapping)
        w_ij.append(np.absolute(weights_ij))
        top_sum = weights_ij * (v_ij - vi_bar)
        num.append(top_sum)
    kappa = np.sum(w_ij)
    ratings = user_to_movie_mapping[user_test].values()
    v_bar_i = (np.sum(list(ratings)))/np.float(len(ratings))
    if kappa==0:
        kappa = 1
    prediction = v_bar_i + (np.sum(num) / kappa)
    if np.isnan(prediction) == True:
        prediction = get_all_user_averages(movie_to_user_mapping)
    if abs(prediction)<1.0:
        prediction = 1
    if abs(prediction)>5.0:
        prediction = 5
    return round(prediction, 2)


user_to_movie_mapping,movie_to_user_mapping = training(train_file)

with open(test_file, "r") as file:
    for i in file:
        movie_test, user, actual_ratings = i.split(",")
        movie_test = int(float(movie_test))
        user = int(float(user))
        actual_ratings = float(actual_ratings)
        pred = predictions(user, movie_test, user_to_movie_mapping, movie_to_user_mapping)
        with open(prediction_file, "a+") as wf:
            wf.write(str(movie_test)+','+str(user)+','+str(pred))
            wf.write("\n")



predictions = pd.read_csv("../code/complete_prediction.txt",sep=",",header=None)
predictions.columns = ["movie_id", "user", "predictions"]
test_data = pd.read_csv(test_file,sep=",",header=None)
test_data.columns = ["movie_id", "user", "original"]
predictions.sort_values(['movie_id', 'user'], ascending=[True, False])
test_data.sort_values(['movie_id', 'user'], ascending=[True, False])
test_data["absolute_diff_error"] = abs(test_data["original"] - predictions["predictions"])
test_data["square_error"] = (test_data["original"] - predictions["predictions"]) ** 2

test_data.to_csv("Results.csv")
mean_abs_error = np.sum(test_data["absolute_diff_error"]) / len(test_data["absolute_diff_error"])
root_mean_square_error = np.sqrt(((np.sum(test_data["square_error"]))/(len(test_data["square_error"]))))

print("MAE",mean_abs_error)
print("RMSE",root_mean_square_error)
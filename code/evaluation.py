import pandas as pd
import numpy as np


predictions = pd.read_csv("../code/complete_prediction.txt",sep=",",header=None)
predictions.columns = ["movie_id", "user", "predictions"]
test_data = pd.read_csv("../data/TestingRatings.txt",sep=",",header=None)
test_data.columns = ["movie_id", "user", "original"]
predictions.sort_values(['movie_id', 'user'], ascending=[True, False])
test_data.sort_values(['movie_id', 'user'], ascending=[True, False])

test_data["Predicted"] = predictions["predictions"]
test_data["absolute_diff_error"] = abs(test_data["original"] - predictions["predictions"])
test_data["square_error"] = (test_data["original"] - predictions["predictions"]) ** 2
test_data.to_csv("Results.csv")

print(test_data["absolute_diff_error"])
print(test_data["square_error"])

mean_abs_error = np.sum(test_data["absolute_diff_error"]) / len(test_data["absolute_diff_error"])
root_mean_square_error = np.sqrt(((np.sum(test_data["square_error"]))/(len(test_data["square_error"]))))

print("MAE",mean_abs_error)
print("RMSE",root_mean_square_error)
exit()




train_file = "D:/Collaborative_Filtering/data/TrainingRatings.txt"
test_file =  "D:/Collaborative_Filtering/data/test.txt"
prediction_file = "complete_predictions.txt"
test_data = pd.read_csv(test_file,sep=",",header=None)
test_data.columns = ["movie_id", "user", "original"]


predictions = pd.read_csv("../code/complete_predictions_test.txt",sep=",",header=None)
predictions.columns = ["movie_id", "user", "predictions"]
test_data = pd.read_csv("../code/TestingRatings.csv",sep=",",header=None)
print(test_data)
#test_data.columns = ["movie_id", "user", "original"]
#predictions.sort_values(['movie_id', 'user'], ascending=[True, False])
test_data.sort_values(['movie_id', 'user'], ascending=[True, False])

test_data["absolute_diff_error"] = abs(test_data["original"] - predictions["predictions"])
test_data["square_error"] = (test_data["original"] - predictions["predictions"]) ** 2

print(test_data["absolute_diff_error"])
print(test_data["square_error"])


"""

predictions = pd.read_csv("D:/Collaborative_Filtering/code/pred.txt",sep=",",header=None)
predictions.columns = ["movie_id", "user", "predictions"]
test_data = pd.read_csv("D:/Collaborative_Filtering/data/TestingRatings.txt",sep=",",header=None)
test_data.columns = ["movie_id", "user", "original"]
predictions.sort_values(['movie_id', 'user'], ascending=[True, False])
test_data.sort_values(['movie_id', 'user'], ascending=[True, False])

test_data["absolute_diff_error"] = abs(test_data["original"] - predictions["predictions"])
test_data["square_error"] = (test_data["original"] - predictions["predictions"]) ** 2
print(test_data["square_error"] )
"""
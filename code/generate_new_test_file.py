import pandas as pd
import numpy as np

pred = pd.read_csv("D:/Collaborative_Filtering/results/pred_2.txt",sep=",",header=None)
pred.columns = ["movie_id", "user", "predictions"]
testing = pd.read_csv("D:/Collaborative_Filtering/results/to_predict_2.txt",sep=",",header=None)
testing.columns = ["movie_id", "user", "original"]

print(len(testing))
t=testing[~testing.isin(pred)].dropna()
t.to_csv("D:/Collaborative_Filtering/results/to_predict_3.txt",sep=",",header=None,index=False)
print(t)
exit()
s1 = pd.merge(pred, t, how='inner', on=['movie_id', 'user'])
s1.dropna(inplace=True)
print(s1)
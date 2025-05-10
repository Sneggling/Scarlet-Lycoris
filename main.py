#-*- coding: utf-8 -*-
import pandas as pd
import joblib
import numpy as np
import sys

selector = joblib.load("output/feature_selector.pkl")
model = joblib.load("output/model.pkl")

file = sys.argv[1]
csv = pd.read_csv(file,header = None,na_values=['nan', ' nan', 'NaN'],)
csv = csv.iloc[:,5:].replace(r'[^\d.+-]', '', regex=True).apply(pd.to_numeric, errors='coerce') .fillna(0)


#Z-Score
mean = csv.mean()
std = csv.std().replace(0,1)
csv = (csv-mean)/std

info = csv.iloc[0:1].values
# if info.shape[0] != 1026 :
#     if info.shape[0] <= 1026:
#         diff = 1026 - info.shape[0]
#         info = np.concatenate([info,info[:diff]])
# print (info)
info = selector.transform(pd.DataFrame(info))

pred = model.predict(info)
pro = model.predict_proba(info)

print(f"result{pred}")
print(f"proba{pro}")
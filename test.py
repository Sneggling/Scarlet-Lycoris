#-*- coding: utf-8 -*-
import pandas as pd
import joblib
selector = joblib.load("output/feature_selector.pkl")
model = joblib.load("output/model.pkl")
test = pd.read_csv("k5test.csv",header = None)
test = selector.transform(pd.DataFrame(test))

pred = model.predict(test)
pro = model.predict_proba(test)

print(pred)
print(pro)

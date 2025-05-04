# -*- coding: utf-8 -*-
import os
#from pyexpat import _Model
from threading import ThreadError;
from tkinter import Y
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import joblib

if __name__ == '__main__':
    Input_Dir = "sample"
    Output_Dir = "output"
    Tree = 390
    TestSize = 0.2
    RamdomState = 42

    data = []
    labels = []

    for filename in os.listdir(Input_Dir):
        if filename.endswith(".csv"):
            label =int( os.path.splitext(filename)[0])
            file_ = Input_Dir+"/"+filename
            df = pd.read_csv(file_, header=None)
            data.append(df)

            num_samples = len(df)
            labels.append(pd.Series([label] * num_samples, name="label"))

        x = pd.concat(data, axis=0).reset_index(drop=True)
        y = pd.concat(labels, axis=0).reset_index(drop=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TestSize, random_state=RamdomState)
    model =  RandomForestClassifier(n_estimators=Tree,n_jobs = -1,class_weight = "balanced",random_state=RamdomState)

    selector = SelectFromModel(estimator = model, threshold="median" )
    x_train_selected = selector.fit_transform(x_train,y_train)
    x_test_selected = selector.transform(x_test)

    model.fit(x_train_selected,y_train)
    test = model.score(x_train_selected,y_train)
    print(f"testscore: {test:.4f}")

    #save
    joblib.dump(selector, os.path.join(Output_Dir, "feature_selector.pkl"))
    
    selected_features = selector.get_support(indices=True)
    pd.DataFrame({"feature_index": selected_features}).to_csv(
        os.path.join(Output_Dir, "selected_features.csv"), 
        index=False
    )
    joblib.dump(model, os.path.join(Output_Dir, "model.pkl"))





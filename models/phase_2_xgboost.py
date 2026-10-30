# Accuracy of the model is: 85.39

from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("clab_data.csv")
X = dataset.drop(columns=["route_id", "image_file"])
y, route_labels = pd.factorize(dataset["route_id"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

xgb_train = xgb.DMatrix(
    X_train,
    label=y_train,
    enable_categorical=False
)
xgb_test = xgb.DMatrix(
    X_test,
    label=y_test,
    enable_categorical=False
)

params = {
    'objective': 'multi:softmax',
    'num_class': len(route_labels),
    'max_depth': 3,
    'learning_rate': 0.1,
}
n=50
model = xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)

preds = model.predict(xgb_test)
accuracy= accuracy_score(y_test,preds)
print('Accuracy of the model is:', accuracy*100)
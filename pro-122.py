import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
x,y=fetch_openml("nnist_784",version=1,return_X_y=True)
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(classes)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=23,train_size=7500,tset_size=2500)
x_train_scaled=x_train/255.0
x_test_scaled=x_test/255.0
clf=LogisticRegretion(solver="saga",multi_class="multinomial").fit(x_train_scaled,y_train)
y_predict=clf.predict(x_test_scaled)
accuracy=accuracy_score(accuracy)
print(accuracy)
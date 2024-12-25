import math
from _ast import In
import lr
import matplotlib
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL.GimpGradientFile import linear
from fontTools.designspaceLib.statNames import BOLD_ITALIC_TO_RIBBI_STYLE
from numpy.ma.core import reshape
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
import  pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import tree


xl = np.array([2019, 2020, 2021, 2022,2023,2024]).reshape(-1,1)
yl = np.array([237091, 220675, 199308, 200204, 227460,158317 ])

futurePredictions = [2025,2026,2027,2028,2029]

model = LinearRegression()
model.fit(xl,yl)
accuracyScore = model.score(xl,yl)
print(accuracyScore)

y_pred = model.predict(xl)
IntegerResult = np.round(y_pred)

x_train, x_test, y_train, y_test = train_test_split(xl,yl,  test_size=0.1)
model.fit(x_train, y_train)
myp2 = model.predict(x_test)
myp2 = np.round(myp2)

arr3 = np.array(list(y_train), dtype='float32')
print('list of converted array long',* np.round(arr3))


plt.bar(futurePredictions, arr3)
plt.xlabel("year")
plt.ylabel("Sales Predictions")
plt.title("TOYOTA PREDICTIVE ANALYSIS")

plt.show()


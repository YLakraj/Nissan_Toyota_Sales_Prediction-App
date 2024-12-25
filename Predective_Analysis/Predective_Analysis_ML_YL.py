import matplotlib
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL.GimpGradientFile import linear
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import  pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import tree


knn = KNeighborsClassifier(n_neighbors=1)
"""year = [2019,2020,2021,2022,2023,2024]
unit_Sold = [123755, 88076, 92567, 70965,85170,55025]"""


xl = [ [2019], [2020], [2021], [2022],[2023],[2024] ]
yl = [ 123755, 88076, 92567, 70965, 85170,55025 ]


"""plt.bar(x, y, color = "blue")
plt.show()

plt.bar(year,unit_Sold,color = "green")
plt.show()"""

nissanData = pd.read_csv('nissan_sales_canada3.csv')
##print(nissanData)
X_Line = nissanData['year']
Y_Line = nissanData['units_sold']
print(nissanData)
##model = DecisionTreeClassifier()
model = LinearRegression()
model.fit(xl,yl)

xnew = [[2025]]
y_pred01 = model.predict(xnew)
print(np.round(y_pred01) )

xnew2 = [[2026]]
y_pred02 = model.predict(xnew2)
print(np.round(y_pred02) )

xnew3 = [[2027]]
y_pred03 = model.predict(xnew3)
print(np.round(y_pred03) )

xnew4 = [[2028]]
y_pred04 = model.predict(xnew4)
print(np.round(y_pred04))

xnew5 = [[2029]]
y_pred05 = model.predict(xnew5)
print(np.round(y_pred05) )

xnew6 = [[2030]]
y_pred06 = model.predict(xnew6)
print(abs(np.round(y_pred06)))

xnew7 = [[2031]]
y_pred07 = model.predict(xnew7)
print(abs(np.round(y_pred07) ))


##test_train split
x_train, x_test, y_train, y_test = train_test_split(xl,yl,  test_size=0.1)
model.fit(x_train, y_train)
myp2 = model.predict(x_test)
score = accuracy_score(y_test, myp2)
print(score)
knn.fit(xl, yl)
plt.scatter(x_test,y_test)
##plt.show()

##linear regression

X_train = xl[:12]
Y_train = yl[:12]
X_test = xl[-4:]
Y_test = yl[-4:]


linear2 = linear_model.LinearRegression()
linear2.fit(X_train,Y_train)
accScore = linear2.score(X_test,Y_test)
print(accScore)


#Coefficient and y-intercept
print('coefficient : \n', linear2.coef_)
print('intercept: \n', linear2.intercept_)

linpred2 = linear2.predict(X_test)
linpred3 = linear2.predict(Y_test)

for x in range(len(linpred2)):
    print(np.round(linpred2[x]), X_test[x], Y_test[x])

##for k in linear2.predict(X_test):
for k in range(2024, 2027):
    mypred = np.round(linear2.predict(X_test))
       #print(k,' ',np.round(linear2.predict(X_test)))

    print(mypred, 'year', k)
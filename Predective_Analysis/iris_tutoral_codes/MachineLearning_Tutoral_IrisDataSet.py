import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import  load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

"""iris = load_iris()
##print(iris)

##print the measurment for each observation
print(iris.data)

##prints the species
print(iris.target)
##prints the names of the diffrent species
print(iris.target_names)
##print dataType value for dataset
print(type(iris.target))
#validate if the data/datalayout is correct 150 diffrent observation in 4 features
print(iris.data.shape)
##shows the responses
print(iris.target.shape)

knn = KNeighborsClassifier(n_neighbors=1)
X = iris.data
Y = iris.target
##insert data in the fit funct,ion to train and predict
Prediction_output = knn.fit(X,Y)
print(Prediction_output )
##making prediction from iris dataset observation

print(knn.predict([ [5.1,3.5,1.4,0.2] ]))
print(knn.predict([ [5.9,3,5.1,1.8] ]))
print(knn.predict([ [2.9,3.1,5.1,8] ]))


##seperate data into train and test group
##xtrain is use to train the computer and x_test is use to test
# the preformance of the computer when making the prediction
##25% of the data is used to make the test and 75% is used for training
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5, random_state=42)
print(X_train.shape)
print(X_test.shape)
knn.fit(X_train,Y_train)

knnprediction = knn.predict(X_test)
print(knnprediction)
print(Y_test)
##show the accuracy percentage of the prediction outcome
preformance = metrics.accuracy_score(Y_test,knnprediction)
print(preformance)

##create an empty list
k_values = {}
##create loop with a cutoff of 25
k = 1
while k <=25:
    knn = KNeighborsClassifier(n_neighbors=k)
    ##train the data as it is being looped through
    knn.fit(X_train, Y_train)
    ##predict data from the x_train
    predictions = knn.predict(X_test)
    ##check the accuracy of the y_test
    preformance = metrics.accuracy_score(Y_test,predictions)
    ##store the performance accuracy in a variable and round 4 decimal places
    k_values[k] = round(preformance,4)
    ##increment the value to +1
    k += 1
    #print the values
print(k_values)
##plot the graph from the values
plt.plot(list(k_values.keys()), list(k_values.values()))
plt.xlabel("k value")
plt.ylabel("performance")
plt.show()"""

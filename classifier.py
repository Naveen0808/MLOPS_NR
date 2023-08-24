import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#import joblib

df = pd.read_csv("/content/MLOPS_NR/data/Iris.csv")

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y = df['Species']

#run = experiment.start_logging()

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size= 0.10, shuffle=True)


#step1: initialise the model class
clf = DecisionTreeClassifier(criterion="gini", max_depth=4)
#step:2 train the model with training data
clf.fit(X_train, y_train)

#step-3 evaluate the model with testing data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy is ..",accuracy*100)
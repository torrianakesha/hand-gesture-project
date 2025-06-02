# Run this program on your local python 
# interpreter, provided you have installed 
# the required libraries. 

# Importing the required packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

# Function to split the dataset 
def splitdataset(data): 
    df = pd.read_csv(data) # Data Frame excel file

    # Separating the target variable 
    X = df.drop('label', axis=1) # kinukuha lahat ng features ng xyz sa columnn except sa label
    Y = df['label'] 

    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100) 
    
    return X, Y, X_train, X_test, y_train, y_test 
    
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): # xtrain - dataset na gagamitin for training

    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) #initialize

    # Performing training 
    clf_gini.fit(X_train, y_train) # ytrain = labels
    return clf_gini 
    
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 

    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 

    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 


# Function to make predictions 
def prediction(X_test, clf_object): # clf object = model

    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) # may function na predict
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
    
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    
    # print("Confusion Matrix: ", 
    #     confusion_matrix(y_test, y_pred)) 
    
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
    
    # print("Report : ", 
    # classification_report(y_test, y_pred)) 

# Funtion for model 
def trainedmodel():
    X, Y, X_train, X_test, y_train, y_test = splitdataset("trainingdata.csv") 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    return clf_gini
    
    
# Driver code 
def main(): 
    
    # Building Phase 
    X, Y, X_train, X_test, y_train, y_test = splitdataset("trainingdata.csv") 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
    
    # Operational Phase 
    print("Results Using Gini Index:") 
    
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
    
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    
# Calling main function 
if __name__=="__main__": 
    main() 
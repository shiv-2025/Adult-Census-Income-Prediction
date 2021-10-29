#In this notebook we will apply different algorithms like Logistic regression, Random forest etc and find out which performs best.
#Based on best model we will create pickle file and save it for deployment

#import required library
import pandas as pd

#get the data
url1 = 'https://github.com/shiv-2025/Adult-Census-Income-Prediction/blob/main/Database/X_cat.csv?raw=true'
X_cat = pd.read_csv(url1)

url2 = 'https://github.com/shiv-2025/Adult-Census-Income-Prediction/blob/main/Database/y.csv?raw=true'
y = pd.read_csv(url2)

print(X_cat.head(5))

#Create train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cat,y, random_state=1234, test_size= 0.2)

#Applying Logistic regression alogorithm to train test and find accuaracy of model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, y_train)

accuracy = round(lr.score(X_test, y_test)*100, 2)

print( 'Accuracy of Decision Logistic regression model : ', accuracy,'%')

#Check with Random forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

rf.fit(X_train, y_train)

accuracy = round(rf.score(X_test, y_test)*100, 2)

print( 'Accuracy of Decision Tree model : ', accuracy,'%')

#get result for test data
y_pred = rf.predict(X_test)

# Saving the model as pickle file
import pickle
pickle.dump(rf, open('salary_clf.pkl', 'wb'))
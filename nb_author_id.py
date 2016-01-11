#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
#https://pypi.python.org/pypi/preprocess/1.1.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

### create classifier
clf = GaussianNB()

### fit the classifier on the training features and labels
t0 = time()
tr = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

### calculate and return the accuracy on the test data

#     Using method in the sklearn.naive_bayes module 
accuracy = clf.score(features_test, labels_test)

#########################################################

print accuracy


### Output from print statements
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#training time: 1.084 s
#prediction time: 0.229 s
#0.973833902162

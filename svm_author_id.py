#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
#########################################################
#########################################################
### your code goes here ###

### create classifier
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
#http://scikit-learn.org/stable/modules/svm.html
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.metrics import accuracy_score

#clf = SVC(kernel="linear")

###########################################################
##  Pt 5: Kernal = rbf, C = 10000.0
###########################################################
#clf = SVC(kernel="rbf", C = 10000.0)
#### fit the classifier on the training features and labels
#t0 = time()
#tr = clf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#
#### use the trained classifier to predict labels for the test features
#t0 = time()
#pred = clf.predict(features_test)
#print "prediction time:", round(time()-t0, 3), "s"
#
#### calculate and return the accuracy on the test data
## Using the sklearn.metrics module
#accuracy = accuracy_score(labels_test, pred)
#
##########################################################
#print 'C: 10000.0'
#print accuracy
###########################################################
###Pt 5 Output
##no. of Chris training emails: 7936
##no. of Sara training emails: 7884
##training time: 125.41 s
##prediction time: 12.975 s
##C: 10000.0
##0.990898748578
###########################################################
###########################################################


##########################################################
#  Pt 2: The following was added in after initial training
##########################################################
#print len(features_train)
#print len(labels_train)

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#print len(features_train)
#print len(labels_train)
##########################################################
##Pt 2 Output
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#training time: 0.138 s
#prediction time: 1.123 s
#0.884527872582
##########################################################
##########################################################

##########################################################
#  Pt 3: Kernal = rbf
##########################################################
#clf = SVC(kernel="rbf")
##########################################################
##Pt 3 Output
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#training time: 0.12 s
#prediction time: 1.196 s
#0.616040955631
##########################################################
##########################################################


### fit the classifier on the training features and labels
#t0 = time()
#tr = clf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#
#### use the trained classifier to predict labels for the test features
#t0 = time()
#pred = clf.predict(features_test)
#print "prediction time:", round(time()-t0, 3), "s"
#
#### calculate and return the accuracy on the test data
## Using the sklearn.metrics module
#accuracy = accuracy_score(labels_test, pred)
#
##########################################################
#
#print accuracy

##########################################################
##Pt 1 Output
### Output from print statements
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#training time: 183.719 s
#prediction time: 19.04 s
#0.984072810011

###########################################################
##  Pt 4: Kernal = rbf, various C's
###########################################################
#C = [10.0, 100., 1000., 10000.]
#
#for c in C:
#    clf = SVC(kernel="rbf", C = c)
#    ### fit the classifier on the training features and labels
#    t0 = time()
#    tr = clf.fit(features_train, labels_train)
#    print "training time:", round(time()-t0, 3), "s"
#    
#    ### use the trained classifier to predict labels for the test features
#    t0 = time()
#    pred = clf.predict(features_test)
#    print "prediction time:", round(time()-t0, 3), "s"
#    
#    ### calculate and return the accuracy on the test data
#    # Using the sklearn.metrics module
#    accuracy = accuracy_score(labels_test, pred)
#
##########################################################
#    print 'C: ',c
#    print accuracy

###########################################################
###Pt 4.1 Output
#training time: 0.125 s
#prediction time: 1.206 s
#C:  10.0
#0.616040955631
###########################################################
###Pt 4.2 Output
#training time: 0.116 s
#prediction time: 1.218 s
#C:  100.0
#0.616040955631
###########################################################
###Pt 4.3 Output
#training time: 0.114 s
#prediction time: 1.269 s
#C:  1000.0
#0.821387940842
###########################################################
###Pt 4.4 Output
#training time: 0.121 s
#prediction time: 0.986 s
#C:  10000.0
#0.892491467577
###########################################################
###########################################################



##########################################################
#  Pt 6: Kernal = rbf, C = 10000. 
# Predictions for various elements
##########################################################
#c = 10000.
#
#clf = SVC(kernel="rbf", C = c)
#### fit the classifier on the training features and labels
#t0 = time()
#tr = clf.fit(features_train, labels_train)
#print "training time:", round(time()-t0, 3), "s"
#
#### use the trained classifier to predict labels for the test features
#t0 = time()
#pred = clf.predict(features_test)
#print "prediction time:", round(time()-t0, 3), "s"
#
#### calculate and return the accuracy on the test data
## Using the sklearn.metrics module
#accuracy = accuracy_score(labels_test, pred)
#
#print 'C: ',c
#print accuracy
#
#
#TE = [10,26,50]
#for e in TE:
#    print e,': ',pred[e]
#

#########################################################

#What class does your SVM (0 or 1, corresponding to Sara and Chris respectively) 
#predict for element 10 of the test set? The 26th? The 50th? 
#(Use the RBF kernel, C=10000, and 1% of the training set. 
#Normally you'd get the best results using the full training set, 
#but we found that using 1% sped up the computation considerably 
#and did not change our results--so feel free to use that shortcut here.)
#
#And just to be clear, the data point numbers that we give here (10, 26, 50) 
#assume a zero-indexed list. So the correct answer for element 
#100 would be found using something like answer=predictions[100]
##########################################################
##Pt 6 Output
#training time: 0.113 s
#prediction time: 1.013 s
#C:  10000.0
#0.892491467577
#10 :  1
#26 :  0
#50 :  1
##########################################################
##########################################################


##########################################################
#  Pt 7: Kernal = rbf, C = 10000. on full training set 
# Count number of predictions for Chris
##########################################################
c = 10000.

clf = SVC(kernel="rbf", C = c)
### fit the classifier on the training features and labels
t0 = time()
tr = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

### calculate and return the accuracy on the test data
# Using the sklearn.metrics module
accuracy = accuracy_score(labels_test, pred)

print 'C: ',c
print accuracy

print 'Chris: ',(pred == 1).sum()
print 'Sarah: ',(pred != 1).sum()

##########################################################
##Pt 7 Output
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#training time: 119.535 s
#prediction time: 11.857 s
#C:  10000.0
#0.990898748578
#Chris:  877
#Sarah:  881
##########################################################
##########################################################

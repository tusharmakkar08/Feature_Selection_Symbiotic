'''
Code for feature selection using Trees Classifier
Author: Tushar Makkar <tusharmakkar08[at]gmail.com>
Date: 22.03.2015
'''
from sklearn.ensemble import ExtraTreesClassifier
import mushrooms_bfgs
import operator

def featureReduction(X, Y):
    '''
    Reduce Features using Tree based feature selection
    X : X input
    Y : Y input
    '''
    clf = ExtraTreesClassifier()
    XNew = clf.fit (X, Y).transform(X)
    featureImportanceList = clf.feature_importances_
    featureImportanceDictionary = {}
    featureCount = 0 
    for i in featureImportanceList : 
        featureImportanceDictionary[featureCount] = i 
        featureCount += 1
    return featureImportanceDictionary
    sortedFeatureDictionary = sorted(featureImportanceDictionary.items(), key=operator.itemgetter(1))
    return sortedFeatureDictionary[1:]
    #~ print len(sortedFeatureDictionary[1:])

def netFeatureImportance(X, Y, N):
    '''
    Gives feature importances of features using tree classifier
    INPUT
    X : Features
    Y : Output
    N : Number of times we run random forest classifier
    '''
    featureImportance = featureReduction(X, Y)
    for i in xrange(N - 1): 
        trialImportance = featureReduction(X, Y)
        for j in trialImportance:
            featureImportance[j] += trialImportance[j]
    for i in featureImportance:
        featureImportance[i] /= N
    sortedFeatureDictionary = sorted(featureImportance.items(), key=operator.itemgetter(1))
    return sortedFeatureDictionary[1:]
    
if __name__ == '__main__':
    (train_X, train_y, test_X, test_y) = mushrooms_bfgs.initialize(float(1)/100)
    print netFeatureImportance(train_X, train_y, 15)

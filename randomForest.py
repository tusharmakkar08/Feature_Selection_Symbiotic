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
    sortedFeatureDictionary = sorted(featureImportanceDictionary.items(), key=operator.itemgetter(1))
    print sortedFeatureDictionary

if __name__ == '__main__':
    (train_X, train_y, test_X, test_y) = mushrooms_bfgs.initialize(float(99)/100)
    featureReduction(train_X, train_y)

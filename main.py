'''
Main Integration class
Author: Tushar Makkar <tusharmakkar08[at]gmail.com>
Date: 16.04.2015
'''

import mushrooms_bfgs, randomForest, Sampling, PGCM, WilcoxonTest
from copy import deepcopy

if __name__ == '__main__':
    (train_X, train_y, test_X, test_y) = mushrooms_bfgs.initialize(float(80)/100)
    N = 22 # Number of Iterations at max can be equal to Number of features - 1
    NumberOfFeaturesRemoved = 0 
    FeaturesRemoved = []
    for i in xrange(1, N + 1):
        # Maximum can remove all the features
        importanceOfRandomForest = randomForest.netFeatureImportance(train_X, train_y, 15)
        importanceOfSymbioticAlgorithm = Sampling.test_initialize()
        #~ print "Feature Importance by Random Forest ", importanceOfRandomForest
        #~ print "Feature Importance by Symbiotic Algorithm ", importanceOfSymbioticAlgorithm
        if len(FeaturesRemoved) == 0 :
            (train_X_Sym, test_X_Sym) = mushrooms_bfgs.featureRemoval(train_X, test_X, [importanceOfSymbioticAlgorithm[i - 1][0]])
            (train_X_For, test_X_For) = mushrooms_bfgs.featureRemoval(train_X, test_X, [importanceOfRandomForest[i - 1][0]])
        else:
            trialFeatureSym = deepcopy(FeaturesRemoved)
            trialFeatureFor = deepcopy(FeaturesRemoved)
            trialFeatureSym.append(importanceOfSymbioticAlgorithm[i - 1][0])
            trialFeatureFor.append(importanceOfRandomForest[i - 1][0])
            print trialFeatureFor, trialFeatureSym
            (train_X_Sym, test_X_Sym) = mushrooms_bfgs.featureRemoval(train_X, test_X, trialFeatureSym)
            (train_X_For, test_X_For) = mushrooms_bfgs.featureRemoval(train_X, test_X, trialFeatureFor)
        wilcoxonOut = WilcoxonTest.WilcoxonTest(train_X.ravel(), train_X_Sym.ravel() ,train_X_For.ravel())
        if wilcoxonOut == 1:
            if importanceOfRandomForest[i - 1][0] in FeaturesRemoved:
                continue
            FeaturesRemoved.append(importanceOfRandomForest[i - 1][0])
            newData_Tr = deepcopy(train_X_For)
            newData_Ts = deepcopy(test_X_For)
        elif wilcoxonOut == 0:
            if importanceOfSymbioticAlgorithm[i - 1][0] in FeaturesRemoved:
                continue
            FeaturesRemoved.append(importanceOfSymbioticAlgorithm[i - 1][0])
            newData_Tr = deepcopy(train_X_Sym)
            newData_Ts = deepcopy(test_X_Sym)
        else:
            break
        #~ print newData_Tr[0]
        NumberOfFeaturesRemoved += 1
    print "Total Number of Features Removed is ", NumberOfFeaturesRemoved
    print "Features Removed are", FeaturesRemoved

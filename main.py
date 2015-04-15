'''
Main Integration class
Author: Tushar Makkar <tusharmakkar08[at]gmail.com>
Date: 16.04.2015
'''

import mushrooms_bfgs, randomForest, Sampling, PGCM, WilcoxonTest
from copy import deepcopy

if __name__ == '__main__':
    (train_X, train_y, test_X, test_y) = mushrooms_bfgs.initialize(float(80)/100)
    N = 5 # Number of Iterations at max can be equal to Number of features - 1
    for i in xrange(1, N + 1):
        # Maximum can remove all the features
        importanceOfRandomForest = randomForest.netFeatureImportance(train_X, train_y, 15)
        importanceOfSymbioticAlgorithm = Sampling.test_initialize()
        #~ print "Feature Importance by Random Forest ", importanceOfRandomForest
        #~ print "Feature Importance by Symbiotic Algorithm ", importanceOfSymbioticAlgorithm
        (train_X_Sym, test_X_Sym) = mushrooms_bfgs.featureRemoval(train_X, test_X, [importanceOfSymbioticAlgorithm[i - 1][0]])
        (train_X_For, test_X_For) = mushrooms_bfgs.featureRemoval(train_X, test_X, [importanceOfRandomForest[i - 1][0]])
        wilcoxonOut = WilcoxonTest.WilcoxonTest(train_X.ravel(), train_X_Sym.ravel() ,train_X_For.ravel())
        if wilcoxonOut == 1:
            newData_Tr = deepcopy(train_X_For)
            newData_Ts = deepcopy(test_X_For)
        elif wilcoxonOut == 0:
            newData_Tr = deepcopy(train_X_Sym)
            newData_Ts = deepcopy(test_X_Sym)
        else:
            break
        train_X = deepcopy(newData_Tr)
        test_X = deepcopy(newData_Ts)
        #~ print train_X[0]

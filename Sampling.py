'''
Helper class for Greedy Sampling 
Author: Tushar Makkar <tusharmakkar08[at]gmail.com>
Date: 5.02.2015
'''

import datareader, mushrooms_bfgs, mlpy, nltk, PGCM

def GreedySampling (PGCM_matrix, oldPGCM, numberOfFeatures, Aggregation):
    '''
    Returns the Sampled Matrix with the Aggregation Function used
    Args:
        PGCM_Matrix : Input PGCM Matrix
        OldPGCM : PGCM matrix with features one less than PGCM input matrix
        numberOfFeatures : total number of features
        Aggregation : Aggregation used for sampling eg : min, max, avg
    Returns:
        The Sampled matrix
    '''
    X = {}
    Y = {}
    score = {}
    for i in xrange(1, numberOfFeatures+1): 
        X[i] = oldPGCM[(i,i)]
        score[i] = 100
        for j in xrange(i+1, numberOfFeatures+1):
            Y[(i,j)] = PGCM_matrix[(i,j)]
            if Aggregation == "min":
                score[i] = min(score[i], PGCM_matrix[(i,j)])
            if Aggregation == "max":
                score[i] = max(score[i], PGCM_matrix[(i,j)])
            if Aggregation == "avg":
                score[i] = score[i]+PGCM_matrix[(i,j)]
                print score[i]
        if Aggregation == "avg":
            score[i] = score[i]/(numberOfFeatures-i+1)
    print score
    
def test_initialize():
    (train_X, train_y, test_X, test_y) = mushrooms_bfgs.initialize(float(99)/100)
    print "Number of Training Data =",len(train_y)
    print "Number of Testing Data =",len(test_y)
    PGCM_0 = PGCM.makePairs(train_X, train_y, test_X, test_y)
    oldPGCM = PGCM_0
    N = 22
    GreedySampling(PGCM_0, oldPGCM, 22, "avg")
    
if __name__ == "__main__":
    test_initialize()


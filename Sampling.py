'''
Helper class for Greedy Sampling 
Author: Tushar Makkar <tusharmakkar08[at]gmail.com>
Date: 5.02.2015
'''

import datareader, mushrooms_bfgs, mlpy, nltk, PGCM

def GreedySampling (PGCM_matrix, OldPGCM, numberOfFeatures, Aggregation):
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
    for i in xrange(1, N+1): 
        x[i] = PGCM_0[(i,i)]
        score[i] = 100
            for j in xrange(1,N+1):
                if i != j:
                    Y[(i,j)] = PGCM_0[(i,j)]
                    score[i] = min(score[i], PGCM_0[(i,j)])
    print score[i]
    
def test_initialize():
    (train_X, train_y, test_X, test_y) = mushrooms_bfgs.initialize(float(99)/100)
    print "Number of Training Data =",len(train_y)
    print "Number of Testing Data =",len(test_y)
    PGCM_0 = PGCM.makePairs(train_X, train_y, test_X, test_y)
    oldPGCM = PGCM_0
    N = 21 
    GreedySampling(PGCM_0, OldPGCM, N,"min")
    
if __name__ == "__main__":
    test_initialize()


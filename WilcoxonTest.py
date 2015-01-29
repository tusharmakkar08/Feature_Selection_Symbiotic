'''
Helper class for checking whether 2 distributions are same or not
Author: Tushar Makkar <tusharmakkar08[at]gmail.com>
Date: 29.01.2015
'''

from scipy import stats

def WilcoxonTest(Original_input, Symbiotic_output, Modified_GA_output):
    '''
    Returns the most similar output to Original_input out of 
    Symbiotic_output and GA_output using Wilcoxon Rank Sum Test.
    Args:
        Original_input: Data with N features
        Symbiotic_output: Data with N-1 features extracted using 
                          symbiotic algorithm
        Modified_GA_output: Data with N-1 features extracted using 
                            Modified Genetic algorithm
    Returns:
        The best suited N-1 features for a given distribution
    '''

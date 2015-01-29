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
        The best suited N-1 features for a given distribution or [-1]
    '''
    z_stat_for_symbiotic, p_val_for_symbiotic = stats.ranksums(
                            Symbiotic_output, Original_input)
    z_stat_for_GA, p_val_for_GA = stats.ranksums(
                            Modified_GA_output, Original_input)
    print p_val_for_GA , p_val_for_symbiotic
    if (p_val_for_GA < 0.5 and p_val_for_symbiotic < 0.5) : 
        return [-1]
    elif (p_val_for_GA > p_val_for_symbiotic): 
        return Modified_GA_output
    else: 
        return Symbiotic_output
        
#~ 
#~ if __name__ == '__main__':
    #~ print WilcoxonTest([1,2,3,4], [1,2,3,5],[0,9,10,11])
#~ 

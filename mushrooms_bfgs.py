import csv, datareader, math, mlpy, matplotlib.pyplot as plt
import numpy, random, scipy.optimize

def get_pca(X, n):
	'''
	Takes X and convert it into n dimensional space and returns the
	answer
	'''
	pca = mlpy.PCA()
	pca.learn(X)
	return  pca.transform(X, k = n)

def plot_data(z, y):
	# Plots the 2 D data 
	plt.set_cmap(plt.cm.Paired)
	fig1 = plt.figure(1)
	title = plt.title("PCA on mushroom dataset")
	plot = plt.scatter(z[:, 0], z[:, 1], c=y)
	labx = plt.xlabel("First component")
	laby = plt.ylabel("Second component") 
	plt.show()


if __name__ == "__main__":
    print "Feature selection on Mushrooms data set"
    raw_input("Press Enter to continue...")
     
    print "Parsing input data..."
    input_file = 'mushrooms_bfgs.data' 
    input_test_file = ''
    custom_delimiter = ',' 
    proportion_factor = float(1)/2
    split = True 
    input_columns = range(1, 23) 
    output_column = 0
    input_literal_columns = [1] * 23
    input_label_mapping = {1:{'b':0, 'c':1, 'x':2, 'f':3, 'k':4, 's':5},
     2:{'f':0, 'g':1, 'y':2, 's':3}, 
     3:{'n':0, 'b':1, 'c':2, 'g':3, 'r':4, 'p':5, 'u':6,
      'e':7, 'w':8, 'y':9}, 
     4:{'t':0, 'f':1}, 
     5:{'a':0, 'l':1, 'c':2, 'y':3, 'f':4, 'm':5, 'n':6, 'p':7, 's':8}, 
     6:{'a':0, 'd':1, 'f':2, 'n':3}, 
     7:{'c':0, 'w':1, 'd':2},
     8:{'b':0, 'n':1},
     9:{'k':0, 'n':1, 'b':2, 'h':3, 'g':4, 'r':5, 'o':6, 'p':7, 'u':8,
      'e':9, 'w':10, 'y':11},
     10:{'e':0, 't':1},
     11:{'b':0, 'c':1, 'u':2, 'e':3, 'z':4, 'r':5, '?':6}, 
     12:{'f':0, 'y':1, 'k':2, 's':3}, 
     13:{'f':0, 'y':1, 'k':2, 's':3},
     14:{'n':0, 'b':1, 'c':2, 'g':3, 'o':4, 'p':5, 'e':6, 'w':7, 'y':8},
     15:{'n':0, 'b':1, 'c':2, 'g':3, 'o':4, 'p':5, 'e':6, 'w':7, 'y':8},
     16:{'p':0, 'u':1},
     17:{'n':0, 'o':1, 'w':2, 'y':3},
     18:{'n':0, 'o':1, 't':2}, 
     19:{'c':0, 'e':1, 'f':2, 'l':3, 'n':4, 'p':5, 's':6, 'z':7},
     20:{'k':0, 'n':1, 'b':2, 'h':3, 'r':4, 'o':5, 'u':6, 'w':7, 'y':8},
     21:{'a':0, 'c':1, 'n':2, 's':3, 'v':4, 'y':5}, 
     22:{'g':0, 'l':1, 'm':2, 'p':3, 'u':4, 'w':5, 'd':6}}
    output_literal = True
    output_label_mapping = {'p':1, 'e':0}
   
    (train_X, train_y, test_X, test_y) = datareader.readInputData(
    input_file, input_test_file, True, custom_delimiter, 
    proportion_factor, split, input_columns, output_column, 
    input_literal_columns, input_label_mapping, output_literal, 
    output_label_mapping)
    no_of_dimension = input("Enter number of dimension you want to "
							"reduce\n ")
    z = get_pca(train_X, no_of_dimension)
    print "Parsing complete!\n"
    print len(train_X[0])
    print len(z[0])
    # Uncomment the following line to plot the training data set
    # plot_data(z, train_y)
   

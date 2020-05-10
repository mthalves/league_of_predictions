import numpy as np
import plot
import preprocessing
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.neural_network import MLPClassifier

# 1. Loading the dataset
print("#####\n# Pre-Processing\n#####")
X, y = preprocessing.get_data(pca=True)
print('| Number of samples:',len(X))
print('| Number of parameters:',len(X[0]))
print('| Number of classes:',len(np.unique(y)))

# 2. Starting the learning process
print("#####\n# Learning Process\n#####")
# a. creating the mlps
classifiers = \
	{'MLP': MLPClassifier(hidden_layer_sizes=(4,8,8,8,4,),activation='relu', solver='adam', max_iter=500),}

# b. training it and predicting
for mlp in classifiers:
	train_sizes, train_scores, test_scores = learning_curve(classifiers[mlp], X, y,\
	 train_sizes=np.array([0.1, 0.25, 0.5, 0.75, 1. ]), cv=5)

	plot.learning_curve(train_sizes,train_scores,test_scores)
	score = cross_validate(classifiers[mlp],X,y,return_train_score=True,cv=5)
	
	print(mlp,np.array(score['train_score']).mean(),np.array(score['test_score']).mean())
	print(mlp,np.array(train_scores).mean(),np.array(test_scores).mean())
import numpy as np
import plot
import preprocessing
from scipy.spatial import distance
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
# - here we add just the best MLP (found via test) 
classifiers = \
	{'MLP': MLPClassifier(hidden_layer_sizes=(4,8,8,8,4,),activation='relu', solver='adam', max_iter=500),}

# b. training it and predicting
for mlp in classifiers:
	# - training and testing to obtain the learning curve
	train_sizes, train_scores, test_scores = learning_curve(classifiers[mlp], X, y,\
	 train_sizes=np.array([0.1, 0.25, 0.5, 0.75, 1. ]), cv=5)
	plot.learning_curve(train_sizes,train_scores,test_scores)

	# - cross validating the method
	score = cross_validate(classifiers[mlp],X,y,return_train_score=True,return_estimator=True,cv=5)
	print(mlp,np.array(score['train_score']).mean(),np.array(score['test_score']).mean())

	# - performing the probabilistic estimation using the best fitted
	# classificator into the cross validation
	best_estimator = score['estimator'][list(score['test_score']).index(max(score['test_score']))]
	y_pred = best_estimator.predict_proba(X)
	y_prob = np.array([np.array([1,0]) if class_ == 0 else np.array([0,1]) for class_ in y])
	y_dist = np.diag(distance.cdist(y_pred,y_prob,'chebyshev'))

	# - plotting the probablistic result
	y_sort = np.sort(y_dist)
	plot.line_graph(np.arange(len(y_sort)),y_sort,'Amostra','Erro','predict_proba.pdf')

	# - plotting confusion matrix
	plot.confusion_matrix(best_estimator,X,y)

# That's all folks... :}
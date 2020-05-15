import matplotlib.pyplot as plt
import numpy as np
import nbformat as nbf
import os
from sklearn.metrics import plot_confusion_matrix

# Open a jupyter notebook to show a dataframe.
# Args: 
# - data_paths: array of string; the paths to the datafiles.
# Return: void.
def jupyter_dataframes(data_paths):
	# 1. Creating the notebook
	notebook = nbf.v4.new_notebook()

	# 2. Building the notebook
	file = open('plots/pretty_dataframe.ipynb', 'w')
	notebook['cells'] = []
	for path in data_paths:
		# a. defining the code
		code = \
"""import pandas as pd

# 1. Loading the data
data = pd.read_csv('"""+path+"""')

# 2. Displaying the dataframe
data"""

		# b. adding new cells to the notebook
		notebook['cells'].append(nbf.v4.new_code_cell(code))

	# 3. Writting the ipynb file
	nbf.write(notebook, file)
	file.close()

	# 4. Runnning the notebook into a new terminal tab
	os.system('gnome-terminal --tab -- jupyter-notebook plots/pretty_dataframe.ipynb')

# Create a bar plot and save it.
# Args:
# - labels: array of strings; bar names.
# - values: array or numpy array; height of the bars.
# - figname: string; name for the saved figure.
# - ylabel: string; label of the y axis; default None.
# - ylim: tuple; limits for y axis; default None,
def bars(labels,values,figname='bars.pdf',ylabel=None,ylim=None):
	# 1. Create bar plot
	fig = plt.figure()

	plt.bar(labels,values,width=0.6,color=['b','r'])

	if ylabel is not None:
		plt.ylabel(ylabel,fontsize='x-large')
	if ylim is not None:
		plt.ylim(ylim)

	for i in range(len(values)):
		plt.text(x = i-0.1, y = values[i]+(0.05*values[i]), s = values[i], size = 18)

	# 2. Saving the figure
	plt.savefig('plots/'+figname,bbox_inches='tight')

# Create a scatter plot of parameter distribution and save it.
# Args:
# - parameters: matrix of data.
# - figname: string; name for the saved figure.
def parameters(parameters,figname='parameters.pdf'):
	fig = plt.figure(figsize=(8,6))

	x = np.array([j for i in range(len(parameters)) for j in range(len(parameters[i])) ])
	y = np.array([parameters[i,j] \
		for i in range(len(parameters)) for j in range(len(parameters[i])) ])
	plt.scatter(x,y,c=np.sqrt(np.abs(y)))

	plt.xlabel('Parâmetro')
	plt.ylabel('Valor')
	plt.savefig('plots/'+figname,bbox_inches='tight')

# Create a line plot of pca explained variance and save it.
# Args:
# - parameters: matrix of data.
# - figname: string; name for the saved figure.
def pca(pca_,figname='pca.pdf'):
	plt.plot(np.cumsum(pca_.explained_variance_ratio_))
	plt.xlabel('Número de componentes',fontsize='x-large')
	plt.ylabel('Informação acumulada',fontsize='x-large')
	plt.savefig('plots/'+figname,bbox_inches='tight')

# Create a line plot of the mlp learning curve and save it.
# Args:
# - train_sizes: array of ints; size of the training sets used to create the learning curve.
# - train_scores: array of floats; score for the training sets.
# - test_scores: array of floats; score for the test sets.
# - figname: string; name for the saved figure.
def learning_curve(train_sizes,train_scores,test_scores,figname='learning_curve.pdf'):
	fig = plt.figure(figsize=(8,4))

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.1,
	                 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.1,
	                 color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	         label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	         label="Cross-validation score")
	plt.legend(loc="best")

	plt.xlabel('Número de Amostras')
	plt.ylabel('Taxa de Acerto')
	plt.savefig('plots/'+figname,bbox_inches='tight')

# Plot and save the confusion matrix.
# Args:
# - estimator: estimator to predict the data.
# - X: matrix of floats; parameters.
# - Y: array of floats; labels.
# - figname: string; name for the saved figure.
def confusion_matrix(estimator,X,y,figname='cm.pdf'):
	plot_confusion_matrix(estimator,X,y,\
		normalize='true',display_labels=['Vitória Time Azul','Vitória Time Vermelho'],\
		cmap=plt.cm.Blues)
	plt.savefig('plots/'+figname,bbox_inches='tight')

# Create a line plot and save it.
# Args:
# - x: array; data x position.
# - y: array; data y position.
# - xlabel: string; x axis label.
# - xlabel: string; y axis label.
# - figname: string; name for the saved figure.
def line_graph(x,y,xlabel,ylabel,figname='line_graph.pdf'):
	fig = plt.figure(figsize=(8,6))
	plt.plot(x, y, '-', color="b")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig('plots/'+figname,bbox_inches='tight')
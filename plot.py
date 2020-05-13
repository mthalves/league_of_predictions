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

# Create a bar plot and save it into a PDF file.
# Args:
# - labels: array of strings; bar names.
# - values: array or numpy array; height of the bars.
# - figname: string; name for the saved figure.
# - ylabel: string; label of the y axis; default None.
# - ylim: tuple; limits for y axis; default None,
def bars(labels,values,figname,ylabel=None,ylim=None):
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

def mlps_test_result():
	for filename in ['mlp_arch_test.txt','mlp_scaled_test.txt','mlp_pca95_test.txt','mlp_pca99_test.txt']:
		print(filename)

		data = {}
		with open(filename,'r') as file:
			for line in file:
				token=line.split(';')
				train_data = np.array([float(t) for t in token[1].split(',')])
				test_data = np.array([float(t) for t in token[2].split(',')])

				data[token[0]] = {}
				data[token[0]]['train'] = {'mean':train_data.mean(),'std':train_data.std()}
				data[token[0]]['test'] = {'mean':test_data.mean(),'std':train_data.std()}

				print('  ',token[0],':\n\ttrain:',\
					data[token[0]]['train']['mean'],data[token[0]]['train']['std'],\
					'\n\ttest:',data[token[0]]['test']['mean'],data[token[0]]['test']['std'])

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

def confusion_matrix(estimator,X,y,figname='cm.pdf'):
	plot_confusion_matrix(estimator,X,y,\
		normalize='true',display_labels=['Vitória Time Azul','Vitória Time Vermelho'],\
		cmap=plt.cm.Blues)
	plt.savefig('plots/'+figname,bbox_inches='tight')

def line_graph(x,y,xlabel,ylabel,figname='graph.pdf'):
	fig = plt.figure(figsize=(8,6))
	plt.plot(x, y, '-', color="b")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig('plots/'+figname,bbox_inches='tight')
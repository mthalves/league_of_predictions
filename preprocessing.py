import numpy as np
import pandas as pd
import plot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def split_data_and_class(dataset,data_index,class_index):
	X, y = dataset.iloc[:,data_index],dataset.iloc[:,class_index]
	return X, y

def class_distribution(y,labels=None,print_=False):
	# 1. Getting the classes
	classes = np.unique(np.array(y))

	# 2. Counting the classes occurences
	count = {}
	for class_ in classes:
		if labels is not None:
			count[labels[class_]] = list(y).count(class_)
			if print_:
				print(labels[class_],':',count[labels[class_]])
		else:
			count[class_] = list(y).count(class_)
			if print_:
				print(class_,':',count[class_])

	# 3. Returning the result
	return count

def write_dataset(X,y,header,filename='dataset.csv'):
	with open(filename,'w') as file:
		# - writing header
		for i in range(len(header)):
			if i == len(header)-1:
				file.write(header[i]+'\n')
			else:
				file.write(header[i]+',')

		# - writing data
		for label, params in zip(y,X):
			file.write(str(label)+',')
			for i in range(len(params)):
				if i == len(params)-1:
					file.write(str(params[i])+'\n')
				else:
					file.write(str(params[i])+',')

def get_data(scale=False,pca=False,show_details=False,open_notebook=False):
	# 1. Loading the original dataset
	print('| Reading the data')
	dataset = pd.read_csv('data/high_diamond_ranked_10min.csv')

	# 2. Pre-processing the data
	# a. splitting parameters and class labels
	X,y = split_data_and_class(dataset,range(2,len(dataset.iloc[0,:])),1)

	# b. analysing class distribution
	if show_details:
		distribution = class_distribution(y,labels={1:'Win',0:'Lose'},print_=True)
		plot.bars(['Vitória Time Azul','Vitória Time Vermelho'], [distribution['Win'],distribution['Lose']],
			'win-lose_bars.pdf',ylabel='Número de Jogos',ylim=(0,6000))

	header = list(X.columns)
	if scale:
		scaler = StandardScaler()
		X = scaler.fit_transform(X,y)

		plot.parameters(X[:100,:])

		header.insert(0,'blueWins')
		write_dataset(X,y,header,'data/pre_processed_high_diamond_ranked_10min.csv')
	elif pca:
		pca_ = PCA(.99)
		X = pca_.fit_transform(X,y)
		selected_comp = [list(clist).index(max(clist)) for clist in pca_.components_]
		print('| | selected components (columns index):',selected_comp)
		print('| | selected components (columns name):',[header[c] for c in selected_comp])
		print('| | n_components:',pca_.n_components_)
		print('| | explained_variance_ratio:',pca_.explained_variance_ratio_)
		print('| | singular_values:',pca_.singular_values_)
		plot.pca(pca_)
	else:
		X = np.array(X)
		y = np.array(y)
		
	# 3. Showing the pre processed data frame and
	# the original data frame into jupyter
	if open_notebook:
		plot.jupyter_dataframes(['../data/high_diamond_ranked_10min.csv',\
						'../data/pre_processed_high_diamond_ranked_10min.csv'])

	# 4. Returning pre-processed data
	return X, y
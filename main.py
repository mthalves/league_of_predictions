#####
# IMPORTS
#####
import matplotlib.pyplot as plt
import nbformat as nbf
import os
import pandas as pd

#####
# PLOT METHODS
#####
def show_dataframe(data_path):
	notebook = nbf.v4.new_notebook()
	code = \
"""import pandas as pd

# 1. Loading the data
data = pd.read_csv('../data/high_diamond_ranked_10min.csv')

# 2. Displaying the dataframe
data"""

	notebook['cells'] = [nbf.v4.new_code_cell(code)]

	with open('plots/pretty_dataframe.ipynb', 'w') as file:
		nbf.write(notebook, file)

	os.system('gnome-terminal --tab --execute jupyter-notebook plots/pretty_dataframe.ipynb')

def plot_bars(labels,values,figname,ylabel=None,ylim=None):
	fig = plt.figure()

	plt.bar(labels,values,width=0.6,color=['b','r'])

	if ylabel is not None:
		plt.ylabel(ylabel,fontsize='x-large')
	if ylim is not None:
		plt.ylim(ylim)

	for i in range(len(values)):
		plt.text(x = i-0.1, y = values[i]+(0.05*values[i]), s = values[i], size = 18)

	plt.savefig('plots/'+figname+'.pdf',bbox_inches='tight')

#####
# MAIN CODE
#####
# 1. Loading the dataset
# a. displaying the dataframe into a jupyter notebook
show_dataframe('data/high_diamond_ranked_10min.csv')

# b. reading the data
print('| Reading the data')
data = pd.read_csv('data/high_diamond_ranked_10min.csv')

# 2. Pre-processing the data
# a. counting the class distribution (0 lose - 1 win)
print('| Counting Win-Lose')
win = list(data.iloc[:,1]).count(1)
lose = len(data) - win
print('> Win:',win,'| Lose:',lose)

# b. plotting the data distribution
plot_bars(['Blue Victory','Red Victory'], [win,lose],
	'win-lose_dist',ylabel='Number of Games',ylim=(0,6000))

# c. splitting the game ids, parameters and class label
gamesId, X, y = data.iloc[:,0], data.iloc[:,2:],data.iloc[:,1]
import matplotlib.pyplot as plt
import nbformat as nbf
import os

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
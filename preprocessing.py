import numpy as np

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
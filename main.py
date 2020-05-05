import pandas as pd
import plot
from preprocessing import *
from sklearn.preprocessing import StandardScaler

# 1. Loading the dataset
print('| Reading the data')
dataset = pd.read_csv('data/high_diamond_ranked_10min.csv')

# 2. Pre-processing the data
# a. splitting parameters and class labels
X,y = split_data_and_class(dataset,range(2,len(dataset.iloc[0,:])),1)

# b. analysing class distribution
distribution = class_distribution(y,labels={1:'Win',0:'Lose'},print_=True)

plot.bars(['Blue Victory','Red Victory'], [distribution['Win'],distribution['Lose']],
	'win-lose_bars.pdf',ylabel='Number of Games',ylim=(0,6000))

# c. getting header information
header = list(X.columns)

# d. scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X,y)

# e. saving the scaled data
header.insert(0,'blueWins')
write_dataset(X,y,header,'data/pre_processed_high_diamond_ranked_10min.csv')

# 3. Showing the pre processed data frame and
# the original data frame into jupyter
plot.jupyter_dataframes(['../data/high_diamond_ranked_10min.csv',\
				'../data/pre_processed_high_diamond_ranked_10min.csv'])

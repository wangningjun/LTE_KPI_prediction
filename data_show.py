# -*- coding: utf-8 -*-
#coding = utf-8

from pandas import read_csv
from matplotlib import pyplot

# load dataset
data = open('data.csv')
dataset = read_csv(data, header=0, index_col=0)
dataset.dropna(inplace=True)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()
from pandas import read_csv
from pandas import concat
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

predict = read_csv('predict.csv')
test_data = read_csv('pollution.csv', header=0, index_col=0)

predict = predict.values
test_data = test_data.values
scaler = MinMaxScaler(feature_range=(0, 1))
pollution = test_data[:,0]
scaled = scaler.fit_transform(pollution[:,np.newaxis])

n_train_hours = 365 * 24
test_data_0 = scaled[n_train_hours+1:]

diff = np.concatenate([predict, test_data_0], axis=1)
print(diff)
pyplot.figure()
i = 1
groups = [0, 1]
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(diff[:, group])
	pyplot.title(i, y=0.5, loc='right')
	i += 1
pyplot.show()

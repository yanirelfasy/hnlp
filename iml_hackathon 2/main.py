from load_tweets import load_dataset
import numpy as np
import re
import random
import pickle
from classifier import Classifier

TRAIN_FACTOR = 0.7

if __name__ == "__main__":
	# Load the data
	X, y = load_dataset()

	dataset_size = len(y)
	train_size = int(TRAIN_FACTOR * dataset_size)


	# indexes = range(len(y))
	# random.shuffle(indexes)

	# X = [X[i] for i in indexes]
	# y = [y[i] for i in indexes]

	# Split to groups
	X_train = X[:train_size]
	X_validation = X[train_size:]
	y_train = y[:train_size]
	y_validation = y[train_size:]


	loaded_classifier = Classifier()


	predicted = loaded_classifier.classify(X_validation)

	print np.mean(predicted == y_validation)


	# dct = {i:0 for i in range(10)}
	# i = 0

	# for pred, real in zip(predicted, y_validation):

	# 	if pred != real:
	# 		if real == 46:
	# 			print "*" * 50
	# 			print X_validation[i]
	# 			print pred, real

	# 		dct[real] += 1


	# 	i += 1

	# for key, val in dct.items():
	# 	print "%d:" % key, val
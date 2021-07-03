"""
===================================================
	 Introduction to Machine Learning (67577)
			 IML HACKATHON, June 2016

			**  Tweets Classifier  **

Auther(s):

===================================================
"""
import pickle
from load_tweets import load_dataset
from PiplineBagOfWords import PiplineBagOfWords

class Classifier(object):

	def __init__(self):
		TRAIN_FACTOR = 0.7
		
		# Load the data
		X, y = load_dataset()

		dataset_size = len(y)
		train_size = int(TRAIN_FACTOR * dataset_size)

		# Split to groups
		X_train = X[:train_size]
		X_validation = X[train_size:]
		y_train = y[:train_size]
		y_validation = y[train_size:]



		self.clf = PiplineBagOfWords()
		self.clf.train(X_train, y_train)

		# with open("classifier.bin", "rb") as fl:
		#	 self.clf = pickle.load(fl)

	def classify(self,X):
		"""
		Recieves a list of m unclassified tweets, and predicts for each one which celebrity posted it.
		:param X: A list of length m containing the tweet's texts (strings)
		:return: y_hat - a vector of length m that contains integers between 0 - 9
		"""

		return self.clf.predict(X)
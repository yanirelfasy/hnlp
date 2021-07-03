from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


from load_tweets import load_dataset
import numpy as np
import re

class PiplineBagOfWords(object):
	"""
	This class implements a classifier base on bag of words to classify personalities tweets.
	"""
	
	def __init__(self):
		"""
		Initializer for the class
		"""

		self.text_clf = Pipeline([
								  ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(2, 6), lowercase=False)),
							 	  ('tfidf', TfidfTransformer()),
								  ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, n_iter=20, random_state=42)),])

		self.clf = None

		emoticons_str = r"""
			(?:
				[:=;] # Eyes
				[oO\-]? # Nose (optional)
				[D\)\]\(\]/\\OpP] # Mouth
			)"""
		 
		regex_str = [
			emoticons_str,
			r'<[^>]+>', # HTML tags
			r'(?:@[\w_]+)', # @-mentions
			r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
			r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
			r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
			r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
			r'(?:[\w_]+)', # other words
			r'(?:\S)' # anything else
			]

		self.tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
		self.emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


	def tokenize(self, s):
		"""
		Splits s according to precompiled regex
		"""

		return self.tokens_re.findall(s)


	def preprocess(self, s, lowercase=False):
		"""
		Recreate the tweet seperated to the different elements it contains 
		for better analyzing
		"""

		tokens = self.tokenize(s)

		toks = []

		emoji = ""

		for val in tokens:
			if len(val) > 1:
				toks.append(val)
				if len(emoji) > 0:
					toks.append("emoji")
					emoji = ""

			elif len(val) == 1 and ord(val) < 128:
				toks.append(val)
				if len(emoji) > 0:
					toks.append("emoji")
					emoji = ""

			else:
				emoji += val

		if len(emoji) > 0:
			toks.append("emoji")
			emoji = ""

		if lowercase:
			tokens = [token if self.emoticon_re.search(token) else token.lower() for token in tokens]


		return " ".join(toks)


	def train(self, X, y):
		"""
		Train the clasifer with Set X (m*1) and labels y (m*1)
		"""

		X = [self.preprocess(x) for x in X]

		self.clf = self.text_clf.fit(X, y)


	def predict(self, X):
		"""
		Classifies a set of samples X (m*1)
		"""

		X = [self.preprocess(x) for x in X]

		return self.clf.predict(X)

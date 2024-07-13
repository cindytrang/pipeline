from abc import ABC, abstractmethod
import numpy as np

class Classifier(ABC):

	@abstractmethod
	def predict(self, sample):
		"""Predict the label of a sample"""
		pass

	@abstractmethod
	def predict_multiple(self, samples):
		"""Predict the label of a sample"""
		pass

	def predict_label(self, text):
		"""predicts the label string of the text"""
		label_index = np.argmax(self.predict(text))
		return self.index_to_label[label_index]

	def predict_label_probability(self, text, label):
		"""returns the probability that a text has a certain label"""
		label_index = self.index_to_label.index(label)
		return self.predict(text)[label_index]

class Dataset(ABC):
	@abstractmethod
	def get_samples(self):
		"""Generator for the content of a Dataset"""
		pass

	@property
	@abstractmethod
	def size(self) -> int:
		"""Number of samples in the Dataset"""
		pass

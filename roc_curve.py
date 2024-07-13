import torch
import matplotlib.pyplot as plt
import torch

from flair.data import Corpus
from flair.models import TextClassifier
from flair.datasets import CSVClassificationCorpus
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from torch.nn.functional import softmax
from flair_util import get_corpus
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

path = "datasets/Corpus/"
dataset_indicator = "Corpus"
corpus: Corpus = get_corpus(path, dataset_indicator)
test_sentences = list(corpus.test)

classifier = TextClassifier.load("path")
classifier.eval()

def embed_sentences(batch_sentences):
    classifier.embeddings.embed(batch_sentences)
    return [sentence.embedding for sentence in batch_sentences]

def create_data_loader(test_sentences, batch_size):
    data_loader = DataLoader(test_sentences, batch_size=batch_size, collate_fn=lambda x: x)
    return data_loader

batch_size = 16  
test_data_loader = create_data_loader(test_sentences, batch_size)

labels, probabilities = [], []
for sentence in test_sentences:
    if sentence.text.strip() == '':
        print("Warning: Skipping an empty sentence.")
        continue
    
    classifier.predict(sentence)
    
    label_dict = sentence.labels[0].to_dict()
    if 'confidence' in label_dict:
        probabilities.append(label_dict['confidence'])  # Probability of the positive class
        labels.append(label_dict['value'])  # The actual label
    else:
        print(f"Warning: Could not make a prediction for sentence: '{sentence.text}'.")

fpr, tpr, _ = roc_curve(labels, probabilities, pos_label='predator')
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

precision, recall, _ = precision_recall_curve(labels, probabilities, pos_label='predator')

# Plot the Precision-Recall curve
plt.figure()
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title('Precision Recall curve')
plt.savefig('precision_recall_curve.png')
plt.show()
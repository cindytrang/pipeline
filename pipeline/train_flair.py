import os
import json
import argparse

# from torch.optim.adam import Adam
from flair.optim import SGDW
from flair.data import Corpus, Sentence
from flair.data import Dictionary, Label
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import AnnealOnPlateau
from pathlib import Path
from flair_util import get_corpus, create_weight_dict
from pathlib import Path
from datetime import datetime, timezone
from argparse import Namespace

# Handler function needed by getTrainArgs taken from the MIT License Copyright (c) 2021 Matthias Vogt
def getTimestamp():
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Handler function for more customisation of model running taken from the MIT License Copyright (c) 2021 Matthias Vogt
def getTrainArgs(default_model_indicator="bert_classifier"):
	parser = argparse.ArgumentParser(description='Train a model')
	parser.add_argument(
		"--model",
		dest='model_indicator',
		default=default_model_indicator, 
		help="the kind of model to use, e.g. bert_classifier or mobilebert_classifier if using modelmaker and bert-large-uncased if using flair",
		required=True
	)
	parser.add_argument(
		"--seq_len",
		dest='seq_len',
		type=int,
		help="maximum sequence length for input text",
		required=True
	)
	parser.add_argument(
		"--dataset",
		dest='dataset_indicator',
		help="which dataset to train on",
		required=True
	)
	parser.add_argument(
		"--project",
		dest='project',
		choices=["modelmaker", "flair"],
		help="which project this model belongs to",
		required=True
	)
	args = parser.parse_args()

	run_settings = vars(args)

	run_settings["run_id"] = '%s__%s_on_%s_with_seq-len-%s' \
		% (getTimestamp(), args.model_indicator, args.dataset_indicator, args.seq_len)

	run_settings["data_dir"] = os.path.join(
		"datasets/%s/" % args.dataset_indicator)

	run_settings["run_dir"] = os.path.join(
		"resources/%s/" % run_settings["run_id"])

	Path(run_settings["run_dir"]).mkdir(parents=True, exist_ok=True)
	with open("%s/run_settings.json" % run_settings["run_dir"], "w") as file:
		json.dump(run_settings, file)

	print("\n---            Run settings            ---")
	for key, val in run_settings.items(): print("%20s: %s" % (key, val))

	return Namespace(**run_settings)

args = getTrainArgs()

# Get the data
corpus: Corpus = get_corpus(args.data_dir, args.dataset_indicator)

# Create a label type so that it can be passed to the classifier 
label_type = 'class'
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=True)

# Create the set weights for the labels
# weights = create_weight_dict(
#     filepath='/dcs/large/u2163087/pipeline/datasets/balanced_training_data.csv',
#     delimiter=',', 
#     label_index=1
# )

# Feature extraction using the transformer document embeddings 
document_embeddings = TransformerDocumentEmbeddings('bert-base-multilingual-cased', fine_tune = True)
# document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune = True)
# document_embeddings = TransformerDocumentEmbeddings('google/electra-base-discriminator', fine_tune=True)
# document_embeddings = TransformerDocumentEmbeddings('bert-base-cased', fine_tune = True)
# document_embeddings = TransformerDocumentEmbeddings('xlnet-base-cased', fine_tune=True)
# document_embeddings = TransformerDocumentEmbeddings(args.model_indicator, fine_tune=True)
# document_embeddings = TransformerDocumentEmbeddings('albert-base-v1', fine_tune = True)

# Initilise the flair classifier with the BERT transformer embeddings 
classifier = TextClassifier(
	document_embeddings,
	label_dictionary=label_dict,
    label_type=label_type, 
)

# Initilise the optimizers for the classifier to use Adam, AdamW and SGDW
# optimizer = Adam(classifier.parameters(), lr=3e-5)
optimizer = SGDW(classifier.parameters(), lr=3e-5, momentum=0.9, weight_decay=1e-5, nesterov=True)

# Initialize the classifier with cpecified optimizer
trainer = ModelTrainer(classifier, corpus)

"""  	
**********  ModelTrainer.train() function parameters that can be included from https://github.com/flairNLP/flair/tree/master **********

        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved
        :param learning_rate: Initial learning rate
        :param mini_batch_size: Size of mini-batches during training
        :param eval_mini_batch_size: Size of mini-batches during evaluation
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits
         until annealing the learning rate
        :param min_learning_rate: If the learning rate falls below this threshold, training terminates
        :param train_with_dev: If True, training is performed using both train+dev data
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch
        :param save_final_model: If True, final model is saved
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
        parameter selection.
        :param num_workers: Number of workers in your data loader.
        :param sampler: You can pass a data sampler here for special sampling of data.
        :param kwargs: Other arguments for the Optimizer
        :return:

		********** example: **********
		# out_dir = os.path.join(args.run_dir, "non_quantized")
		# trainer.train(out_dir,
		# 	learning_rate=3e-5,
		# 	mini_batch_size=8,
		# 	mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
		# 	max_epochs=3, 
		# 	monitor_train=True,
		# 	monitor_test=True
		# )

"""

# Model training
out_dir = os.path.join(args.run_dir, "non_quantized")
trainer.train(out_dir,
	learning_rate= 3e-5,
	mini_batch_size=16,
    anneal_factor=0.5,    
    patience=3,
	mini_batch_chunk_size=4,
	max_epochs=5, 
    optimizer=optimizer,  
    train_with_dev=False,
	monitor_train=True,
	monitor_test=True,
    checkpoint=True,
    anneal_with_restarts=True
)
    
# Saving the trained model for the classification and prediction scores use case
model_path = os.path.join(out_dir, "final-model.pt")
classifier.save(model_path)
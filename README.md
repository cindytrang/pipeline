## Project Overview ##
**Title:** Child Abuse Behavior Detection on Smartphones  

This project aims to develop and improve methods for the early detection of potential child abuse behaviours through the analysis of smartphone communications,
usually very short messages. To demonstrate the alarming system in action and the chat message evaluation process, please watch the demo video: 
https://www.youtube.com/watch?v=2zDAT3vREFY.

## Table of Contents ##
- [Background](#background)
- [Repository Structure](#repository-structure)
- [Thresholds and Labels](#thresholds-and-labels)
- [Installation and Usage](#installation-and-usage)
- [Licenses](#licenses)


## Background ##
This codebase integrates adaptations from "Early Detection of Sexual Predators in Chats" by Matthias Vogt, which is available under the MIT License (see LICENSE.txt). Data used in this project was collected by Mathias Gatti in 2020 and is also licensed under the MIT License (see "LICENSE (data).txt"). Significant enhancements were made through brainstorming and collaboration with the BF-PSR-Framework project on GitHub, which employs a rule-based approach.

## Repository Structure ##
### Folders
- **train_flair**: Hosts all model training code, focusing on corpus handling and classifier initialization.
- **flair_util**: Provides utility scripts for model fine-tuning, utilized by the `train_flair.py` script.
- **add_predictions**: Central script where data analysis components are integrated. This script assigns data stage annotations to provide risk and warning levels.
- **laptopenv/myenv**: Those directories contain environment setups where all project dependencies are installed using pip and conda. See `environment.yml` and `requirements.txt` for details.
- **batch_scripts**: Scripts and command examples for model training and execution, tailored for use with the University of Warwick’s batch compute system. [Batch Compute Guide](https://warwick.ac.uk/fac/sci/dcs/intranet/user_guide/batch_compute)
- **resources folder**: Includes some models results and the model classifiers see some runs with final-model.pt. Using those classifiers is possible with combination to add_preditction.py file.
- **log_files**: Includes example logs and results from model performances that were trained (shows the time of training)
- **Dataset**: All data combinations created and collected throughout the project (e.g., PAN12, Sexual Variations, PJ). Includes scripts for data processing and extraction.


## Additional Files Scripts ##
- **roc_curve.py**: Generates visualizations of the receiver operating characteristic curve.
- **concrete-classes.py**: Utility script for classifier training.
- **ner.py**: Tags phrases within the text data; customizable and not used in the final rule-based model.
- **imagenet_classes**: Supports the image classifier, Alexnet.
- **requirements/environment files**: Logs the libraries used, such as numpy, flair, and pytorch versions.


## Thresholds and Labels
The demonstration visualizer is not included in the repository. However, it can be adapted from the project [Chat Visualizer](https://gitlab.com/early-sexual-predator-detection/chat-visualizer) by incorporating additional data linked to message stages. The following thresholds, based on testing with the PJ datasets, help classify predatory chats early on and can be adjusted as needed:

- **Gathering Information and Selecting the Victim**
  - Threshold: 0.55, Count Limit: 5
- **Trust Development and Establishing Credibility**
  - Threshold: 0.6, Count Limit: 3
- **Priming and Desensitizing the Target**
  - Threshold: 0.5, Count Limit: 1
- **BERT Prediction**
  - Threshold: 0.5 (high chance of predatory if the score is above)
- **Risk Window Size**
  - 6
- **Count Limit of Warnings**
  - 3

## Installation and Usage ##
Please refer to the environment.txt (newenv) and requirement.txt(env) files for instructions on setting up the development environment. 
Use the batch script examples in the `batch_scripts` folder to configure the models for the University of Warwick's infrastructure or other devices.

Example command:
```bash
python add_predictions.py --eval_mode segments --window_size 124 --model_version non_quantized --run_id [SPECIFY_RUN_ID]
```
      [SPECIFY_RUN_ID] For example to run the best-performing model use: 
      --run 2024-04-04_14-04-31__bert_classifier_on_PAN12_with_seq-len-512

In the case of wanting to train the classifier with an adjusted classifier, use the example command:

```bash
python train_flair.py --dataset Corpus --project flair --seq_len 512 --model bert_classifier
```

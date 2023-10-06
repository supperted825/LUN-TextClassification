# Unreliable News Classification

This project explores the use of various text classification models and techniques to discern between different types of unreliable and reliable news. Namely, logistic regression, simple neural networks and transformers are used to approach this multi-class classification problem. A combination of TF-IDF and/or transformer embeddings are used as input features to increase the classification performance of the model, which is measured by macro-F1 score.

This work was undertaken as a group project as part of the requirements for CS4248 Natural Language Processing at the National University of Singapore.

## Files

We use Python 3.8 with PyTorch v1.9. Additional core dependencies are transformers v4.26.1, sentencepiece v0.1.95 and nlpaug v1.1.11.

* [train.py](train.py) - Main python file for finetuning huggingface transformer models on the LUN dataset.

* [ablation.py](ablation.py) - Trains a simple NN on TF-IDF inputs only, as part of ablation studies.

* [augment.py](augment.py) - Applies train-test split and text augmentations with NLPaug to yield an augmented training set.

* [src/models.py](./src/models.py) - Transformer-based and simple NNs for text classification.

* [src/loss.py](./src/loss.py) - Various classification loss functions (eg. focal loss).

* [src/learning_rate.py](./src/learning_rate.py) - Functions for layerwise learning rate decay and lowered backbone learning rate.

* [src/utils.py](./src/utils.py) - Utility functions to be called during training routines.

* [notebooks/exploration.ipynb](./notebooks/exploration.ipynb) - Notebook for dataset exploration.

* [notebooks/baselines.ipynb](./notebooks/baselines.ipynb) - Notebook for training classical ML baselines (eg. logistic regression)

* [notebooks/analysis.ipynb](./notebooks/analysis.ipynb) - Notebook for analysis and consolidation of training results.

* [requirements.txt](requirements.txt) - Pip-installable requirements for running the project.




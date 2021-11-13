READ ME
------------------------------

## Directory structure

This directory contains these files:
- code: contains iPython notebooks related to this project and its HTML output
	+ data: contains saved data preprocessing results
	o preset
	o augmented
	+ 0_data_preparation.ipynb: data preparation steps, the outputs are saved to "data" folder
	+ 1_data_exploration.ipynb
	+ 2_classification.ipynb: model training and validation using default dataset
	+ 2_classification_augmented.ipynb: model training and validation using augmented dataset
- proposal.pdf: submitted Capstone project proposal

Quora question pair dataset download link: https://www.kaggle.com/c/quora-question-pairs/data

GloVe word vector: http://nlp.stanford.edu/data/glove.840B.300d.zip

Note: this project is created using Python 3.5

## Required Packages

- numpy
- pandas
- seaborn
- matplotlib
- nltk
- spacy
- string
- re
- scipy.stats
- csv
- json
- zipfile
- os
- keras
- time
- datetime
- sklearn
- wordcloud
- tqdm

## Deep learning model results

### Word embedding only
1. adadelta optimizer result - mse as loss function
    - loss = 0.1240, accuracy = 0.8247, f1-score = 0.7487 **BEFORE AUGMENTATION**
    - loss = 0.1414, accuracy = 0.8060, f1-score = 0.7890 **AFTER AUGMENTATION**

2. adam optimizer result - mse as loss function
    - loss = 0.1254, accuracy = 0.8245, f1-score = 0.7492  **BEFORE AUGMENTATION**
    - loss = 0.1357, accuracy = 0.8133, f1-score = 0.7936  **AFTER AUGMENTATION**

3. adam optimizer result - binary cross entropy as loss function
    - loss = 0.4725, accuracy = 0.8211, f1-score = 0.7488 **BEFORE AUGMENTATION**
    - loss = 0.4881, accuracy = 0.8119, f1-score = 0.7870 **AFTER AUGMENTATION**

### Custom 1: concatenate shared features to each embedding output
1. adadelta optimizer - mse as loss function
    - loss = 0.1252, accuracy = 0.8245, f1-score = 0.7508 **BEFORE AUGMENTATION**
    - loss = 0.1420, accuracy = 0.8038, f1-score = 0.7825 **AFTER AUGMENTATION**

### Custom 2: concatenate individual features to each question; concatenate shared features to shared lstm output
1. adadelta optimizer - binary cross entropy as loss function - sigmoid on dense layer
    - loss = 0.5343, accuracy = 0.7161, f1-score = nan **BEFORE AUGMENTATION**
    - loss = 0.5432, accuracy = 0.7130, f1-score = 0.6812 **AFTER AUGMENTATION**
     
2. adam optimizer - binary cross entropy as loss function - sigmoid on dense layer
    - loss = 0.5510, accuracy = 0.7241, f1-score = nan **BEFORE AUGMENTATION**
    - loss = 0.5573, accuracy = 0.7136, f1-score = 0.6760 **AFTER AUGMENTATION**
    
3. adam optimizer - mse as loss function - sigmoid on dense layer
    - loss = 0.1803, accuracy = 0.7327, f1-score = nan **BEFORE AUGMENTATION**
    - loss = 0.1855, accuracy = 0.7279, f1-score = 0.7047 **AFTER AUGMENTATION**
    
4. adam optimizer - mse as loss function - relu on dense layer
    - loss = 0.6883, accuracy = 0.6663, f1-score = nan **BEFORE AUGMENTATION**
    - is not retrained using augmented data since the performance is not good
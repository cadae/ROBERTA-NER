# NER task fine-tuned on pre-trained model allenai/biomed_roberta_base

Test set results:
Accuracy: 0.8846153846153846
F1: 0.8727588201272412
Precision: 0.9002403846153846
Recall: 0.8846153846153846

## Pre-processing

## Training

## Pipeline usage

from pipeline import EntityClassifier

classifier = EntityClassifier('./saved_model')
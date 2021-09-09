# NER task fine-tuned on pre-trained model allenai/biomed_roberta_base

Test set results:

Accuracy: 0.8846153846153846

F1: 0.8727588201272412

Precision: 0.9002403846153846

Recall: 0.8846153846153846

Python notetbook version of the code available under the models dir

## Training and Testing

Run train.py to train on training dataset and save the trained model

Run test.py to test the trained model against the testing dataset

## Pipeline usage

```
from pipeline import EntityClassifier

classifier = EntityClassifier('./saved_model') # point to the dir of the saved model
classifier.classify("Hello World")
```
output is a list of tuples that looks like this
```
[('Hello', 'O'), ('World', 'O')]
```

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import RobertaForTokenClassification, RobertaTokenizer

class NERModel:
    def __init__(self, model_id, num_labels=3):
        self.model_id = model_id
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained('allenai/biomed_roberta_base')
        self.model = RobertaForTokenClassification.from_pretrained(self.model_id, num_labels=self.num_labels)

    def save_model(self, model_dir):
        if self.model:
            self.model.save_pretrained(model_dir)
    
    def word_encodings(self, sentence):
        if self.tokenizer:
            encodings = self.tokenizer(sentence['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
            labels = torch.tensor(sentence['ner_tags'] + [0] * (self.tokenizer.model_max_length - len(sentence['ner_tags'])))
            return { **encodings, 'labels': labels }


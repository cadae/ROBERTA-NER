import torch

from model import NERModel

# the classifer is using ROBERTA tokenizer which has a different way of tokenizing words
# compared to the training and testing dataset.
# in order to achieve the same accuracy as the test set, the input str should be tokenised using
# the same tokenizer as the one that is used for perparing the dataset.

class EntityClassifier:
    def __init__(self, model_id):
        self.ner = NERModel(model_id)
        self.id2label = {
            0: 'O',
            1: 'B',
            2: 'I'
        }
        self.label2id = { value: key for key, value in self.id2label.items() }

    def classify(self, input):
        inputs = self.ner.tokenizer(input, return_tensors="pt")
        tokens = self.ner.tokenizer.tokenize(input)
        tokens = [token.replace("Ġ", "") for token in tokens]
        labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)
        outputs = self.ner.model(**inputs, labels=labels)
        logits = outputs.logits
        probs = logits.softmax(dim=-1).tolist()
        pred_labels = []
        for prob in probs[0][1:len(probs)-2]: # remove start and end tokens
            pred_labels.append(self.id2label[prob.index(max(prob))])
        if len(pred_labels) == len(tokens):
            return list(zip(tokens, pred_labels))
        else:
            print(tokens, pred_labels)
            raise Exception('Incorrect len of tokens')


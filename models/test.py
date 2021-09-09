import torch

from model import NERModel
from train import load_sentences_from_file, convert_to_tensor

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if __name__ == "__main__":
    batch_size = 6
    model_dir="./saved_model/"
    id2label = {
        0: 'O',
        1: 'B',
        2: 'I'
    }
    label2id = { value: key for key, value in id2label.items() }
    ner = NERModel(model_dir, len(id2label))
    ner.model = ner.model.eval().to(ner.device)
    dataset = {}
    dataset['test'] = convert_to_tensor(list(map(ner.word_encodings, load_sentences_from_file('./NERdata/test.tsv',label2id))))
    test_data = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size)

    for batch_index, batch_data in enumerate(test_data):
        with torch.no_grad():
            batch_data = { key: value.to(ner.device) for key, value in batch_data.items() }
            outputs = ner.model(**batch_data)
        
        seq_len = batch_data['attention_mask'].sum(dim=1)
        for index, length in enumerate(seq_len):
            groundtrue = batch_data['labels'][index][:length]
            preds = torch.argmax(outputs[1], dim=2)[index][:length]


    print("Accuracy: {0}".format(accuracy_score(groundtrue.cpu(), preds.cpu())))
    print("F1: {0}".format(f1_score(groundtrue.cpu(),preds.cpu(),average='weighted')))
    print("Precision: {0}".format(precision_score(groundtrue.cpu(),preds.cpu(),average='weighted',zero_division=1)))
    print("Recall: {0}".format(recall_score(groundtrue.cpu(),preds.cpu(),average='weighted')))
import numpy as np

import torch
import torch.optim as optim

from model import NERModel

def load_sentences_from_file(path, label2id):
    sentences = []
    sentence = {
        'tokens': [],
        'ner_tags': []
    }
    for line in open(path, 'r', encoding='utf-8'):
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence['tokens']) > 0:
                sentences.append(sentence)
            sentence = {
                'tokens': [],
                'ner_tags': []
            }
        else:
            splits = line.split()
            if len(splits) == 2:
                sentence['tokens'].append(splits[0])
                sentence['ner_tags'].append(label2id[splits[1]])
    if len(sentence['tokens']) > 0:
        sentences.append(sentence)
    return sentences

def convert_to_tensor(data):
    for entry in data:
        for key, value in entry.items():
            entry[key] = torch.tensor(value)
    return data

if __name__ == "__main__":
    model_id="allenai/biomed_roberta_base"
    model_dir="./saved_model/"
    id2label = {
        0: 'O',
        1: 'B',
        2: 'I'
    }
    label2id = { value: key for key, value in id2label.items() }
    ner = NERModel(model_id, len(id2label))
    ner.model.config.id2label = id2label
    ner.model.config.label2id = label2id

    dataset = {}
    dataset['train'] = convert_to_tensor(list(map(ner.word_encodings, load_sentences_from_file('./NERdata/train.tsv',label2id))))
    ner.model.train().to(ner.device)
    optimizer = optim.AdamW(params=ner.model.parameters(), lr=1e-5)

    epochs = 3
    batch_size = 6
    # since we don't want to print the loss every iteration
    log_freq = 100
    train_data = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size)

    # start training
    print("Start training")
    for epoch in range(epochs):
        cached_loss = 0
        for batch_index, batch_data in enumerate(train_data):
            # load batch data in memory
            batch_data = { key: value.to(ner.device) for key, value in batch_data.items() }
            # compute logits and loss
            outputs = ner.model(**batch_data)
            loss = outputs.loss
            # backpropagation
            loss.backward()
            # update model
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()
            # cache loss for plotting
            cached_loss += loss.item()
            if batch_index % log_freq == 0 and batch_index > 0:
                # cache losses in final list
                print("Loss: %f" % (cached_loss / (log_freq * batch_size)))
                cached_loss = 0
        # update model at the end of each epoch
        optimizer.step()
        optimizer.zero_grad()

    ner.save_model(model_dir)

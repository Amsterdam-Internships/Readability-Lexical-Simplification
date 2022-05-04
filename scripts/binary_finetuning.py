import pandas as pd
import random
import torch
from transformers import BertTokenizerFast
from transformers import BertForPreTraining
from transformers import AdamW
import os
import argparse
import torch.optim as optim
from tqdm import tqdm

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def create_input(normal, simple):
    normal_sents = normal['sentence'].tolist()
    simple_sents = simple['sentence'].tolist()

    sentence_a = []
    sentence_b = []
    label = []

    i = 0
    for normal_sent, simple_sent in zip(normal_sents, simple_sents):
        i+=1
        if i>10000:
            break
        random_number = random.random()
        if random_number > 0.5:
            sentence_a.append(normal_sent)
            sentence_b.append(simple_sent)
            label.append(1)
        else:
            sentence_a.append(simple_sent)
            sentence_b.append(normal_sent)
            label.append(0)
    return sentence_a, sentence_b, label

def create_labels(inputs,label):
    # Creating labels for NSP
    inputs['next_sentence_label'] = torch.LongTensor([label]).T
    inputs['labels'] = inputs.input_ids.detach().clone()

    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []
    for i in range(inputs.input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    return inputs

def run_training(model, optim, loader, device):
    epochs = 2
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=False)
        for batch in loop:
            print("batch")
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            print("after input to device")
            token_type_ids = batch['token_type_ids'].to(device)
            print("after token_type_ids to device")
            attention_mask = batch['attention_mask'].to(device)
            print("after attention_mask to device")

            next_sentence_label = batch['next_sentence_label'].to(device)
            print("after NSL to device")

            labels = batch['labels'].to(device)
            print("after labels to device")

            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            next_sentence_label=next_sentence_label,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


def main():
    parser = argparse.ArgumentParser()

    # Directory of evaluation data (BenchLS/ Lexmturk/ NNSeval)
    # #ToDo Fill in the exact datasets
    parser.add_argument("--lr",
                        default=5e-5,
                        type=int,
                        required=False,
                        help="The learning rate")
    args = parser.parse_args()

    print("Load model and tokenizer")
    model = BertForPreTraining.from_pretrained('bert-large-uncased-whole-word-masking')
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased-whole-word-masking')

    print("Loading in data")
    normal = pd.read_csv("../datasets/Wikipedia simple/normal.aligned", sep="\t",
                         names=["subject", "nr", "sentence"])
    simple = pd.read_csv("../datasets/Wikipedia simple/simple.aligned", sep="\t",
                         names=["subject", "nr", "sentence"])

    sentence_a, sentence_b, label = create_input(normal, simple)
    tokenized_sentences = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                       max_length=250, truncation=True, padding='max_length')

    inputs = create_labels(tokenized_sentences, label)
    dataset = OurDataset(inputs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda')
    model.to(device)

    model.train()

    optim = AdamW(model.parameters(), args.lr)
    run_training(model, optim, loader, device)

    output_dir = '../models/finetuned_on_gpu'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

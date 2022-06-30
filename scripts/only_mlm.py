from transformers import BertTokenizer, BertForMaskedLM, BertForPreTraining
import torch
from transformers import AdamW
import pandas as pd
import os
import random
import argparse
import logging
from tqdm import tqdm  # for our progress bar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def create_inputs(tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', max_length=250, truncation=True, padding='max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()
    rand = torch.rand(inputs.input_ids.shape)

    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
               (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    return inputs

def get_Dutch_data(num_sents, level):
    with open ("../datasets/sentences_"+level+".txt", "r") as infile:
        data = infile.readlines()
        data = [i.strip() for i in data][:num_sents]
        print(data)
        text = data
        return text

def get_English_data(num_sents):
    simple = pd.read_csv("../datasets/Wikipedia simple/simple.aligned", sep="\t",
                         names=["subject", "nr", "sentence"])
    simple = simple.sample(frac=1, random_state=1)
    text = simple["sentence"].tolist()[:num_sents]
    return text


def main(my_args=None):

    if my_args is None:
        logger.info("begin")

        parser = argparse.ArgumentParser()

        parser.add_argument("--nr_sents",
                            default=10000,
                            type=int,
                            required=False,
                            help="Number of sentences")

        parser.add_argument("--lr",
                            default=5e-6,
                            type=float,
                            required=False,
                            help="The learning rate")

        parser.add_argument("--epochs",
                            default=2,
                            type=float,
                            required=False,
                            help="The number of epochs")

        parser.add_argument("--model_directory",
                            type=str,
                            required=True,
                            help="Directory to store model")

        parser.add_argument("--random_seed",
                            type=int,
                            default=3,
                            required=False,
                            help="Directory to store model")

        parser.add_argument("--language",
                           default="English",
                           type=str,
                           required=True,
                           help="To finetune the English or the Dutch model")

        parser.add_argument("--level",
                           default="Accepted",
                           type=str,
                           required=False,
                           help="The level of the dutch texts")

        args = parser.parse_args()

        num_sents = args.nr_sents
        nr_epochs = args.epochs
        learning_rate = args.lr
        model_dir = args.model_directory
        seed = args.random_seedu
        language = args.language
        level = args.level

    else:
        num_sents = my_args[0]
        nr_epochs = my_args[1]
        learning_rate = my_args[2]
        model_dir = my_args[3]
        seed = my_args[4]
        language = my_args[5]
        level = my_args[6]

    if language == "English":
        logger.info("you are training an English model")
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')

    if language == "Dutch":
        logger.info("you are training a Dutch model")
        tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
        model = BertForMaskedLM.from_pretrained('GroNLP/bert-base-dutch-cased')

    random.seed(seed)
    torch.manual_seed(seed)

    if language == "English":
        text = get_English_data(num_sents)

    if language == "Dutch":
        text = get_Dutch_data(num_sents, level)

    # text = get_data(num_sents)
    inputs = create_inputs(tokenizer, text)
    dataset = OurDataset(inputs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    device = torch.device('cuda')
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=learning_rate)


    for epoch in range(nr_epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
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

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    main()
import logging
import itertools

import finetuning
import BERT_for_LS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

learning_rates = [5e-5, 5e-6, 5e-7, 5e-8]
num_sents = [1000, 10000, 50000]
evaluation_data = ["../datasets/NNSeval.txt", "../datasets/BenchLS.txt", "../datasets/lex.mturk.txt"]

configurations = list(itertools.product(learning_rates, num_sents))

random_seed = 3

num_epochs = 2

i = 0
for lr, num_sent in configurations:
    i += 1
    logger.info(f"Starting configuration{i}: lr:{lr}, {num_sent} sentences")

    model_name = f'FT_lr{lr}_{num_sent}sents'
    model_dir = f'../models/{model_name}'

    logger.info(f"Starting fine tuning")

    finetuning.main([num_sent,
                     num_epochs,
                     lr,
                     model_dir,
                     random_seed])

    for data in evaluation_data:
        logger.info(f"Starting evaluation on {data}")
        BERT_for_LS.main([model_dir,
                          data,
                          250,
                          10,
                          False])

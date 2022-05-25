import logging
import itertools

import finetuning
import BERT_for_LS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

num_sent = 10000
evaluation_data = "../datasets/NNSeval.txt"
lr = 5e-6
random_seeds = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
num_epochs = 2

i = 0
for seed in random_seeds:
    i += 1
    logger.info(f"Starting configuration{i}: seed:{seed}")

    model_name = f'FT_seed{seed}_lr{lr}'
    model_dir = f'../models/{model_name}'

    logger.info(f"Starting fine tuning")

    finetuning.main([num_sent,
                     num_epochs,
                     lr,
                     model_dir,
                     seed])

    logger.info(f"Starting evaluation on {evaluation_data}")
    BERT_for_LS.main([model_dir,
                      evaluation_data,
                      250,
                      10,
                      False])

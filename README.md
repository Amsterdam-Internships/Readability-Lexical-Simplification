# Readability : Contextualized Lexical Simplification

This repository contains all code that has been used for my Master thesis on Contextualized Lexical Simplification.

![img.png](media/img.png)

[comment]: <> (![]&#40;media/examples/emojis.png&#41;)

---

## Project Folder Structure

There are the following folders in the structure:

1) [`scripts`](./scripts): Folder containing the used scripts
1) [`datasets`](./datasets): Folder containing the used datasets
1) [`results`](./results): Folder containing the produced results
1) [`models`](./models): Folder where used embedding models can be stored
1) [`tests`](./tests): Test examples
1) [`media`](./media): Folder containing media files (icons, video)

---


## Installation

In order to run all steps of the lexical simplification pipeline, follow these steps:

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/Readability-Lexical-Simplification
    ```
1) Install all dependencies:

    pip install -r requirements.txt
    ```
---


## Deploying the pipeline

Simplifications can be made on English and Dutch. They require a number of files:

**Steps Needed for Running the model for English** 
1) Download a word embedding model from (fasttext) and store it in the models folder as __crawl-300d-2M-subword.vec__
1) Download the BenchLS, NNSeval and lex.mturk datasets from https://simpatico-project.com/?page_id=109 dataset and store them in the models folder

Then, the model can be run as follows:
```
python3 BERT_for_LS.py --model bert-large-uncased-whole-word-masking --eval_dir ../datasets/Dutch/dutch_data.txt 
```


**Steps Needed for Running the model for Dutch**
1) Download the word embedding model from https://dumps.wikimedia.org/nlwiki/20160501/ and store it in the models folder as __wikipedia-320.txt__

Then the model can be run as follows:
```
python3 BERT_for_LS.py --model GroNLP/bert-base-dutch-cased --eval_dir ../datasets/Dutch/dutch_data.txt
```

**Additional Arguments can be passed:**

|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--model`| str| `the name of the model that is used for generating the predictions: a path to a folder or a huggingface directory.`|  -|
|`--eval_dir`| str| `path to the file with the to-be-simplified sentences.`| -|
|`--analysis`| Bool| `whether or not to output all the generated candidates and the reason for their removal `|False|
|`--ranking`| Bool| `whether or not to perform ranking of the generated candidates`|False|
|`--evaluation`| Bool| `whether or not to perform an evaluation of the generated candidates`|True|
|`--num_selections`| int| `the amount of candidates to generate`|10|
---

## Finetuning a Model

**Requirements for fine-tuning**
English:
1) Download the simple--regular aligned wikipedia corpus
2) Download the simple wikipedia corpus

Dutch:
1) Download wablieft corpus
2) Download domain-specific data

Can be done in three ways:  
1) Masked language modelling: 
   ```
   python3 only_mlm.py  --nr_sents 10000   
                        --epochs 2
                        --model_directory ../models/MLM_model
                        --seed 3
                        --language nl
                        --level simple
   ```
1) Masked language modelling and next token prediction:
   ```
   python3 mlm_nsp.py   --nr_sents 10000   
                        --epochs 2
                        --model_directory ../models/MLM_model
                        --seed 3
                        --language nl
   ```
1) Masked language modelling and simplification prediction:
   ```
   python3 finetuning.py   --nr_sents 10000   
                           --epochs 2
                           --model_directory ../models/MLM_model
                           --seed 3
   ```





## Notebooks
Notebooks for analyses


## Acknowledgements


This code is based on the LSBert pipeline: https://github.com/qiang2100/BERT-LS

The file "dutch frequencies" is the processed version of SUBTLEX NL (http://crr.ugent.be/programs-data/subtitle-frequencies/subtlex-nl)


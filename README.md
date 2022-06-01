# Readability : Contextualized Lexical Simplification

This repository contains all code that has been used for my Master thesis on Contextualized Lexical Simplification. The repository, like my thesis is a work in progress.

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
    ```bash
    pip install -r requirements.txt
    ```
---


## Usage

Simplifications can be made on English and Dutch. They require a number of files:

**Deploying the pipeline for English** 
1) Download a word embedding model from (fasttext) and store it in the models folder as __crawl-300d-2M-subword.vec__
1) Download the BenchLS, NNSeval and lex.mturk datasets from https://simpatico-project.com/?page_id=109 dataset and store them in the models folder

**Deploying the pipeline for Dutch**
1) Download the word embedding model from https://dumps.wikimedia.org/nlwiki/20160501/ and store it in the models folder as __wikipedia-320.txt__


Then the model can be run as follows:
```
python3 BERT_for_LS.py --model GroNLP/bert-base-dutch-cased --eval_dir ../datasets/Dutch/dutch_data.txt
```

|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--model`| str| `the name of the model that is used for generating the predictions: a path to a folder or a huggingface directory.`|  -|
|`--eval_dir`| str| `path to the file with the to-be-simplified sentences.`| -|
|`--results_file`|  str | `path to file where the performance report is written out`| -|
|`--analysis`| Bool| `whether or not to output all the generated candidates and the reason for their removal `|False|
|`--ranking`| Bool| `whether or not to perform ranking of the generated candidates`|False|
|`--evaluation`| Bool| `whether or not to perform an evaluation of the generated candidates`|True|

|---|:---:|:---:|:---:|

---

## Finetuning a Model

Can be done in one of two ways: todo explain

## Generating Dutch Dataset Sentence
Todo clean and commit

## How it works


## Acknowledgements


Don't forget to acknowledge any work by others that you have used for your project. Add links and check whether the authors have explicitly stated citation preference for using the DOI or citing a paper or so. 
For example:

[comment]: <> (Our code uses [YOLOv5]&#40;https://github.com/ultralytics/yolov5&#41; [![DOI]&#40;https://zenodo.org/badge/264818686.svg&#41;]&#40;https://zenodo.org/badge/latestdoi/264818686&#41;)
This code is based on the LSBert pipeline: https://github.com/qiang2100/BERT-LS

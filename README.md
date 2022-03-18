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
   ```

1) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---


## Usage

Simplifications can be made on English and Dutch. They require a number of files:

**English** 
1) Download a word embedding model from (fasttext) and store it in the models folder as __crawl-300d-2M-subword.vec__
1) Download the BenchLS dataset and store it in the models folder

```
$ python LSBert.py --language eng --eval_dir ../datasets/NNSeval.txt --output_SR_file ../results/aaa
```

**Dutch**
1) Download the word embedding model from https://dumps.wikimedia.org/nlwiki/20160501/ and store it in the models folder as __wikipedia-320.txt__


Then the model can be run as follows:
```
$ python LSBert.py  --eval_dir ../datasets/small_example_dutch.txt --word_embeddings ../models/wikipedia-320.txt --word_frequency datasets/dutch_frequencies.txt  --output_SR_file results/aaa
```

[comment]: <> (|Argument | Type or Action | Description | Default |)

[comment]: <> (|---|:---:|:---:|:---:|)

[comment]: <> (|`--eval_dir`| str| `path to evaluation data.`|  -|)

[comment]: <> (|`--word_embeddings`| str| `path to word emedding model.`| -|)

[comment]: <> (|`--word_frequency`|  str | `path to frequency file`| -|)

[comment]: <> (|`--output_SR_file`| str| `path to results file`|-|)

[comment]: <> (|...|...|...|...|)

---


## How it works

You can explain roughly how the code works, what the main components are, how certain crucial steps are performed...

---
## Acknowledgements


Don't forget to acknowledge any work by others that you have used for your project. Add links and check whether the authors have explicitly stated citation preference for using the DOI or citing a paper or so. 
For example:

[comment]: <> (Our code uses [YOLOv5]&#40;https://github.com/ultralytics/yolov5&#41; [![DOI]&#40;https://zenodo.org/badge/264818686.svg&#41;]&#40;https://zenodo.org/badge/latestdoi/264818686&#41;)
This code is based on the LSBert pipeline: https://github.com/q
iang2100/BERT-LS

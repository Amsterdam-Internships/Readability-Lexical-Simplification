# Readability : Contextualized Lexical Simplification

This repository contains all code that has been used for my Master thesis on Contextualized Lexical Simplification. The repository, like my thesis is a work in progress.


The *_How it works_* section below would contain more technical details for curious people.
If applicable, you can also show an example of the final output.

![](media/examples/emojis.png)

---


## Project Folder Structure

There are the following folders in the structure:

1) [`resources`](./resources): Random nice resources, e.g. [`useful links`](./resources/README.md)
1) [`src`](./src): Folder for all source files specific to this project
1) [`scripts`](./scripts): Folder with example scripts for performing different tasks (could serve as usage documentation)
1) [`tests`](./tests) Test example
1) [`media`](./media): Folder containing media files (icons, video)
1) ...

---


## Installation

Explain how to set up everything. 
Let people know if there are weird dependencies - if so feel free to add links to guides and tutorials.

A person should be able to clone this repo, follow your instructions blindly, and still end up with something *fully working*!

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/Readability-Lexical-Simplification
    ```

1) If you are using submodules don't forget to include `--recurse-submodules` to the step above or mention that people can still do it afterwards:
   ```bash
   git submodule update --init --recursive
   ```

1) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---


## Usage

For running the Dutch Experiment (the first that has been implemented), the following steps are required:

1) Download a word embedding model from https://github.com/clips/dutchembeddings
1) And store it in the models folder

Then the model can be run as follows:
```
$ python LSBert_Dutch.py  --eval_dir datasets/small_example_dutch.txt --word_embeddings models/wikipedia-320.txt --word_frequency datasets/dutch_frequencies.txt  --output_SR_file results/aaa
```

|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--eval_dir`| str| `path to evaluation data.`|  -|
|`--word_embeddings`| str| `path to word emedding model.`| -|
|`--word_frequency`|  str | `path to frequency file`| -|
|`--output_SR_file`| str| `path to results file`|-|
|...|...|...|...|


Alternatively, as a way of documenting the intended usage, you could add a `scripts` folder with a number of scripts for setting up the environment, performing training in different modes or different tasks, evaluation, etc (thanks, [Tom Lotze](https://www.linkedin.com/in/tom-lotze/)!)

---


## How it works

You can explain roughly how the code works, what the main components are, how certain crucial steps are performed...

---
## Acknowledgements


Don't forget to acknowledge any work by others that you have used for your project. Add links and check whether the authors have explicitly stated citation preference for using the DOI or citing a paper or so. 
For example:

Our code uses [YOLOv5](https://github.com/ultralytics/yolov5) [![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


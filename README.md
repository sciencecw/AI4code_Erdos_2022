# AI4Code
#### A Coding Challenge to match Markdown comments to code in Jupyter notebook

"The goal of this competition is to understand the relationship between code and comments in Python notebooks. You are challenged to reconstruct the order of markdown cells in a given notebook based on the order of the code cells, demonstrating comprehension of which natural language references which code."


### Table of Contents
- Notebook 1: A code snippet to reconstruct Jupyter notebooks from the JSON file input

- Notebook 2: A test of Assymetric Syntax Tree (AST) that allows us to parse the python code and extract crucial features 

- Notebook 3: A test of NLP cleaning methods 

- Notebook 4: Preprocessing step. This Notebook does most of the heavy lifting and provides most of the annotations. It defines a python decorator method of implement a pipeline. The main functions are:
  - skims and divides the data
  - processes text with traditional NLP methods
  - apply AST and find out about potential inheritance information
  - pair cells to make final matrix ready for classifiers
  - creates custom validation set, as the validation and training samples must not share entries from the same jupyter notebook

- Notebook 5: Machine Learning Classifiers and plot and metric of result

- Notebook X: A demonstration of the metric used in the competition and make sense of it

Data is not included in this repo, as it is 2GB large. Instructions to download can be found here:
https://www.kaggle.com/c/AI4Code

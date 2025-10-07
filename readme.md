# Joint Representations of Natural Language and Complex Queries

## Quickstart
To start a local server for hosting the embedding visualization website run `python app.py`

To only calculate the embeddings and cache them use `python app.py --precalculate`

## Overview
This repository consists of a CLIP inspired model, which is used to compare Natural Language Queries with Boolean Language Queries. It also contains helper functions for preprocessing data from different sources, and training/evaluating different models.

### app.py
By running this file a local server which hosts a website showing embeddings of given data in a 2D plot

### eval.ipynb
This notebook contains examples for evaluating models with our evaluation functions

### plotting.ipynb
This notebook was used to create plots from data saved to W&B while training models

### train.ipynb
A notebook for training/fine-tuning models with HuggingFace libraries
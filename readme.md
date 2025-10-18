# Joint Representations of Natural Language and Complex Queries

This repository contains a CLIP/SigLIP inspired model for comparing data of two different text modalities (here boolean language queries and natural language queries but adaptable to other types of text). There also is a visualization tool for visualizing the embedding space through Umap.

## Quickstart
To start a local server for hosting the embedding visualization website run `python app.py`

To only calculate the embeddings and cache them use `python app.py --precalculate`

## Overview
This repository consists of a CLIP inspired model, which is used to compare Natural Language Queries with Boolean Language Queries. It also contains helper functions for preprocessing data from different sources, and training/evaluating different models.

### app.py
By running this file a local server which hosts a website showing embeddings of given data in a 2D plot

By adjusting variables in the beginning of this file the input data and used model can be modified.

### eval.ipynb
This notebook contains examples for evaluating models with our evaluation functions

### plotting.ipynb
This notebook was used to create plots from data saved to W&B while training models

### train.ipynb
A notebook for training/fine-tuning models with HuggingFace libraries

## Images

### Visualization Tool
<img width="700" alt="VisOverview" src="https://github.com/user-attachments/assets/505da5f9-8ef2-4c37-a018-3d4f5cf68335" />

### Some Training Results

<img width="700" alt="image" src="https://github.com/user-attachments/assets/0eff1d0e-d3a0-4533-9c31-6a3832a7dea4" />
<img width="700" alt="image" src="https://github.com/user-attachments/assets/7299256f-4823-4094-a8e2-76ffaae5738d" />

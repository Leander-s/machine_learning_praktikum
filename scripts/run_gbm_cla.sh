#!/bin/bash

conda activate ml

cd /machine_learning_praktikum/code

# bace

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/bace.csv gcn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/bace.csv mpnn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/bace.csv gat cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/bace.csv attentivefp cla

# bbbp

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/bbbp.csv gcn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/bbbp.csv mpnn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/bbbp.csv gat cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/bbbp.csv attentivefp cla

# hiv

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/hiv.csv gcn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/hiv.csv mpnn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/hiv.csv gat cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/hiv.csv attentivefp cla

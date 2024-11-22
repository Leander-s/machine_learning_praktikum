#!/bin/bash

conda activate ml

cd /machine_learning_praktikum/code

# freesolv

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv gcn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv mpnn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv gat reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv attentivefp reg

# esol

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv gcn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv mpnn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv gat reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv attentivefp reg

# lipop

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv gcn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv mpnn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv gat reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv attentivefp reg

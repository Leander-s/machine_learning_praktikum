#!/bin/bash

conda activate ml

cd /machine_learning_praktikum/code

# clintox

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/clintox.csv gcn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/clintox.csv mpnn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/clintox.csv gat cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/clintox.csv attentivefp cla

# sider

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/sider.csv gcn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/sider.csv mpnn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/sider.csv gat cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/sider.csv attentivefp cla

# tox21

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/tox21.csv gcn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/tox21.csv mpnn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/tox21.csv gat cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/tox21.csv attentivefp cla

# toxcast

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/toxcast.csv gcn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/toxcast.csv mpnn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/toxcast.csv gat cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/toxcast.csv attentivefp cla

# muv

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/muv.csv gcn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/muv.csv mpnn cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/muv.csv gat cla
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/muv.csv attentivefp cla

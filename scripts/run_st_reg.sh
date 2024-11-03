#!/bin/bash

conda activate ml

cd /machine_learning_praktikum/code

# freesolv

# gnns

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv gcn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv mpnn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv gat reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv attentivefp reg

# dnns

conda run -n ml python svm.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv reg
conda run -n ml python xgb.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv reg
conda run -n ml python dnn_torch.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv reg
conda run -n ml python rf.py /machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv reg

# esol

# gnns

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv gcn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv mpnn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv gat reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv attentivefp reg

# dnns

conda run -n ml python svm.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv reg
conda run -n ml python xgb.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv reg
conda run -n ml python dnn_torch.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv reg
conda run -n ml python rf.py /machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv reg

# lipop

# gnns

conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv gcn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv mpnn reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv gat reg
conda run -n ml python gnn.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv attentivefp reg

# dnns

conda run -n ml python svm.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv reg
conda run -n ml python xgb.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv reg
conda run -n ml python dnn_torch.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv reg
conda run -n ml python rf.py /machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv reg

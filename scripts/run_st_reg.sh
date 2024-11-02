#!/bin/bash
# freesolv

# gnns

python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv gcn reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv mpnn reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv gat reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv attentivefp reg

# dnns

python machine_learning_praktikum/code/svm.py machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv reg
python machine_learning_praktikum/code/xgb.py machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv reg
python machine_learning_praktikum/code/dnn_torch.py machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv reg
python machine_learning_praktikum/code/rf.py machine_learning_praktikum/data/dataset_used_for_modeling/freesolv.csv reg

# esol

# gnns

python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv gcn reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv mpnn reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv gat reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv attentivefp reg

# dnns

python machine_learning_praktikum/code/svm.py machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv reg
python machine_learning_praktikum/code/xgb.py machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv reg
python machine_learning_praktikum/code/dnn_torch.py machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv reg
python machine_learning_praktikum/code/rf.py machine_learning_praktikum/data/dataset_used_for_modeling/esol.csv reg

# lipop

# gnns

python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv gcn reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv mpnn reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv gat reg
python machine_learning_praktikum/code/gnn.py machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv attentivefp reg

# dnns

python machine_learning_praktikum/code/svm.py machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv reg
python machine_learning_praktikum/code/xgb.py machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv reg
python machine_learning_praktikum/code/dnn_torch.py machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv reg
python machine_learning_praktikum/code/rf.py machine_learning_praktikum/data/dataset_used_for_modeling/lipop.csv reg

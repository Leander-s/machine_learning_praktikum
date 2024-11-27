#!/bin/bash

conda activate ml

cd /machine_learning_praktikum/code

# clintox

./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/clintox.csv gcn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/clintox.csv mpnn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/clintox.csv gat cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/clintox.csv attentivefp cla

# sider

./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/sider.csv gcn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/sider.csv mpnn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/sider.csv gat cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/sider.csv attentivefp cla

# tox21

./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/tox21.csv gcn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/tox21.csv mpnn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/tox21.csv gat cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/tox21.csv attentivefp cla

# toxcast

./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/toxcast.csv gcn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/toxcast.csv mpnn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/toxcast.csv gat cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/toxcast.csv attentivefp cla

# muv

./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/muv.csv gcn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/muv.csv mpnn cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/muv.csv gat cla
./run_gbm.sh /machine_learning_praktikum/data/dataset_used_for_modeling/muv.csv attentivefp cla

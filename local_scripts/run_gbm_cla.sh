#!/bin/bash

conda activate ml

cd ../code

# bace

./run_gbm.sh ../data/dataset_used_for_modeling/bace.csv gcn cla
./run_gbm.sh ../data/dataset_used_for_modeling/bace.csv mpnn cla
./run_gbm.sh ../data/dataset_used_for_modeling/bace.csv gat cla
./run_gbm.sh ../data/dataset_used_for_modeling/bace.csv attentivefp cla

# bbbp

./run_gbm.sh ../data/dataset_used_for_modeling/bbbp.csv gcn cla
./run_gbm.sh ../data/dataset_used_for_modeling/bbbp.csv mpnn cla
./run_gbm.sh ../data/dataset_used_for_modeling/bbbp.csv gat cla
./run_gbm.sh ../data/dataset_used_for_modeling/bbbp.csv attentivefp cla

# hiv

./run_gbm.sh ../data/dataset_used_for_modeling/hiv.csv gcn cla
./run_gbm.sh ../data/dataset_used_for_modeling/hiv.csv mpnn cla
./run_gbm.sh ../data/dataset_used_for_modeling/hiv.csv gat cla
./run_gbm.sh ../data/dataset_used_for_modeling/hiv.csv attentivefp cla

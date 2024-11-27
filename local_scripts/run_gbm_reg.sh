#!/bin/bash

conda activate ml

cd ../code

# freesolv

./run_gbm.sh ../data/dataset_used_for_modeling/freesolv.csv gcn reg
./run_gbm.sh ../data/dataset_used_for_modeling/freesolv.csv mpnn reg
./run_gbm.sh ../data/dataset_used_for_modeling/freesolv.csv gat reg
./run_gbm.sh ../data/dataset_used_for_modeling/freesolv.csv attentivefp reg

# esol

./run_gbm.sh ../data/dataset_used_for_modeling/esol.csv gcn reg
./run_gbm.sh ../data/dataset_used_for_modeling/esol.csv mpnn reg
./run_gbm.sh ../data/dataset_used_for_modeling/esol.csv gat reg
./run_gbm.sh ../data/dataset_used_for_modeling/esol.csv attentivefp reg

# lipop

./run_gbm.sh ../data/dataset_used_for_modeling/lipop.csv gcn reg
./run_gbm.sh ../data/dataset_used_for_modeling/lipop.csv mpnn reg
./run_gbm.sh ../data/dataset_used_for_modeling/lipop.csv gat reg
./run_gbm.sh ../data/dataset_used_for_modeling/lipop.csv attentivefp reg

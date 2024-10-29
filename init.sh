#! /bin/bash

conda init

echo "Creating virtual environment"
conda create -y -n ml_gnn_env python=3.6.5

echo "Installing pytorch"
conda install -y -n ml_gnn_env pytorch==1.3.1 cudatoolkit==9.2 -c pytorch -c nvidia

echo "Installing dgl"
conda run -n ml_gnn_env python -m pip install dgl==0.4.1

echo "Installing scikit-learn"
conda install -y -n ml_gnn_env scikit-learn==0.20.1

echo "Installing hyperopt"
conda run -n ml_gnn_env python -m pip -q install bson==0.1.0
conda run -n ml_gnn_env python -m pip -q install hyperopt==0.2

echo "Installing xgboost"
conda install -y -n ml_gnn_env xgboost=0.80 -c conda-forge

echo "Installing rdkit"
conda install -y -n ml_gnn_env -c rdkit rdkit=2019.09.1

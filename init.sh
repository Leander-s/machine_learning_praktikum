#! /bin/bash

conda init

echo "Creating virtual environment"
conda create -y -n ml_gnn_env python=3.6.5

echo "Installing pip"
conda install -y -n ml_gnn_env pip
echo "Installing pandas"
conda install -y -n ml_gnn_env pandas

echo "Installing pytorch"
conda install -y -n ml_gnn_env pytorch==1.3.1 cudatoolkit==9.2 -c pytorch -c nvidia

echo "Installing dgl"
conda run -n ml_gnn_env python -m pip install dgl==0.4.1

echo "Installing scikit-learn"
conda install -y -n ml_gnn_env scikit-learn==0.20.1

echo "Installing sgboost"
conda install -y -n ml_gnn_env conda-forge::xgboost==0.80

echo "Installing rdkit"
conda install -y -n ml_gnn_env -c rdkit rdkit=2019.09.1

echo "Installing hyperopt"
conda run -n ml_gnn_env python -m pip install hyperopt==0.2

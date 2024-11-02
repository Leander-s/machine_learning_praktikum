#! /bin/bash

conda init

echo "Creating virtual environment"
conda create -y -n ml python=3.6.5

echo "Installing pytorch"
conda install -y -n ml pytorch==1.3.1 cudatoolkit==9.2 -c pytorch -c nvidia

echo "Installing dgl"
conda run -n ml python -m pip install dgl-cu92==0.4.1

echo "Installing scikit-learn"
conda install -y -n ml scikit-learn==0.20.1

echo "Installing hyperopt"
conda run -n ml python -m pip -q install hyperopt==0.2
conda run -n ml python -m pip -q uninstall bson
conda run -n ml python -m pip -q install pymongo

echo "Installing xgboost"
conda install -y -n ml xgboost=0.80 -c conda-forge

echo "Installing rdkit"
conda install -y -n ml -c rdkit rdkit=2019.09.1

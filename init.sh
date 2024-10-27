#! /bin/bash

conda init

echo "Creating virtual environment"

conda create -y -n ml_gnn_env python=3.9

echo "Checking for updates"

conda update -y -n ml_gnn_env setuptools

echo "Installing pytorch"

conda install -y -n ml_gnn_env pytorch==2.4.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing dgl"
conda install -y -n ml_gnn_env -c dglteam/label/th24_cu118 dgl

echo "Upgrading numpy"
conda install -y -n ml_gnn_env numpy==2.0.0

echo "Installing scikit-learn"
conda install -y -n ml_gnn_env scikit-learn

echo "Installing hyperopt"
conda install -y -n ml_gnn_env conda-forge::hyperopt

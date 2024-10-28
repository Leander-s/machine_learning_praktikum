#! /bin/bash

conda init

echo "Creating virtual environment"
conda create -y -n ml_gnn_env python=3.9

echo "Installing packaging"
conda install -y -n ml_gnn_env packaging

echo "Checking for updates"
conda update -y -n ml_gnn_env setuptools

echo "Installing pandas"
conda install -y -n ml_gnn_env pandas

echo "Installing pytorch"
conda install -y -n ml_gnn_env pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 torchdata -c pytorch -c nvidia

echo "Installing dgl"
conda install -y -n ml_gnn_env -c dglteam/label/th24_cu121 dgl
conda install -y -n ml_gnn_env -c dglteam/label/th24_cu121 dgl[graphbolt]

echo "Installing scikit-learn"
conda install -y -n ml_gnn_env scikit-learn

echo "Installing hyperopt"
conda install -y -n ml_gnn_env conda-forge::hyperopt

echo "Installing pydantic"
conda install -y -n ml_gnn_env conda-forge::pydantic

echo "Installing cuda tools"
conda install -y -n ml_gnn_env cudatoolkit==12.1
conda install -y -n ml_gnn_env cudnn

#! /bin/bash

echo "Activating virtual environment"

source ./env/bin/activate

echo "Installing dgl"
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html

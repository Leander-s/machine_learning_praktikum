# Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models 

# TODO

Get descriptor-based models to work

# Code

Source code provided in the paper.

## Ubuntu

Nvidia docker toolkit is required:

    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Alternatively you could install miniconda

    https://docs.anaconda.com/miniconda/miniconda-install/

and then:

    ./init.sh
    conda activate ml

## Docker

### Linux / WSL

Build docker

    ./build_docker.sh

Run all the tests in the docker container in the background

    ./run_docker.sh

### Other

Build docker image

    docker build -t ml_container .

Run docker container

    docker run --gpus all -v ./results:/machine_learning_praktikum/code/stat_res -it ml_container

Run all tests from within docker container

    /bin/bash /machine_learning_praktikum/scripts/run_everything.sh



## Activate python environment

    conda activate ml


## Dependencies

- nvidia docker toolkit OR miniconda
- CUDA capable GPU

### Python packages
Should be installed by init.sh

## Execution

### gnn

    python gnn.py ../data/<dataset>/<data>.csv <model> <task>

If that doesn't work, try:

    python3 gnn.py ../data/<dataset>/<data>.csv <model> <task>

- **dataset**: "dataset_used_for_modeling" or "washed_dataset"
- **data**: can be any csv file in the dataset folder
- **model**: "gcn", "mpnn", "gat" or "attentivefp"
- **task**: "cla" or "reg"

# Data

- csv files containing datasets for training/validating/testing.
- output files in .bin format are also stored here

# Figure

Results provided by the paper.

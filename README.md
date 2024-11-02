# TODO


# Code

Source code provided in the paper. Requires CUDA!

## Ubuntu

Nvidia docker toolkit is required:

    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Alternatively you could install miniconda

    https://docs.anaconda.com/miniconda/miniconda-install/

and then:

    ./init.sh
    conda activate ml

## Docker

Build docker image

    docker build -t ml_container .

Run docker container

    docker run --gpus all -it ml_container


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

- **dataset**: "dataset_used_for_modeling" or "washed_dataset"
- **data**: can be any csv file in the dataset folder
- **model**: "gcn", "mpnn", "gat" or "attentivefp"
- **task**: "cla" or "reg"

# Data

- csv files containing datasets for training/validating/testing.
- output files in .bin format are also stored here

# Figure

Results provided by the paper.

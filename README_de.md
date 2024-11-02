# TODO


# Code

Source code der im paper gegeben war. Funktioniert nur mit CUDA! 

## Ubuntu

Das Nvidia docker toolkit muss installiert sein: 

    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Alternativ kann man miniconda installieren

    https://docs.anaconda.com/miniconda/miniconda-install/

und dann einfach:

    ./init.sh
    conda activate ml

## Docker

Build docker image

    docker build -t ml_container .

Run docker container

    docker run --gpus all -it ml_container


## Python environment aktivieren

    conda activate ml


## Dependencies

- nvidia docker toolkit ODER miniconda
- CUDA

### python packages
sollten alle durch das Dockerfile installiert sein

## Ausführung

### gnn

    python gnn.py ../data/<dataset>/<data>.csv <model> <task>

Wenn das nicht funktioniert, probier:

    python3 gnn.py ../data/<dataset>/<data>.csv <model> <task>

- **dataset** "dataset_used_for_modeling" oder "washed_dataset"
- **data** kann irgendein csv file aus dem dataset folder sein
- **model** "gcn", "mpnn", "gat" oder "attentivefp"
- **task** "cla" oder "reg"

# Data

Daten zum trainieren/testen der Modelle.

# Figure

Tabelle der Ergebnisse

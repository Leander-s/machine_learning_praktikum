# TODO


# Code

Source code der im paper gegeben war.

## Ubuntu

Das Nvidia docker toolkit muss installiert sein: 

    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Alternativ kann man miniconda installieren

    https://docs.anaconda.com/miniconda/miniconda-install/

und dann einfach:

    ./init.sh
    conda activate ml

## Docker

### Linux / WSL

Build docker

    ./build_docker.sh

Führe alle Tests im Container im Hintergrund aus

    ./run_docker.sh

### Anderes

Build docker image

    docker build -t ml_container .

Starte Container

    docker run --gpus all -v ./results:/machine_learning_praktikum/code/stat_res -it ml_container

Starte alle Tests im Container

    /bin/bash /machine_learning_praktikum/scripts/run_everything.sh


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

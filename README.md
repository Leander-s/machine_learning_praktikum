# TODO


# Code

Source code der im paper gegeben war. Funktioniert nur mit Nvidia GPUs! 

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

sollten alle durch das Dockerfile installiert sein

## Ausführung

### gnn

    python gnn.py ../data/<dataset>/<data>.csv <model> <task>

- **dataset** kann "dataset_used_for_modeling" oder "washed_dataset" sein
- **data** kann irgendein csv file aus dem dataset folder sein
- **model** kann "gcn", "mpnn", "gat" oder "attentivefp" sein
- **task** kann "cla" oder "reg" sein

### Fragen

Wie genau funktioniert der code. Welche argumente machen was, wo kommt der output raus.

# Data

Data zum trainieren/testen der Modelle.

### Fragen

Was ist der unterschied zwischen "dataset_used_for_modeling" und "washed_dataset"

# Figure

Tabelle der Ergebnisse

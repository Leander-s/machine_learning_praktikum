# TODO


# Code

Source code der im paper gegeben war.

## Docker

Build docker image

    docker build -t ml_container .

Run docker container

    docker run -it ml_container


## Python environment aktivieren

    conda activate ml_gnn_env


## Dependencies

- DGL : https://www.dgl.ai/pages/start.html

sollten alle durch das Dockerfile installiert sein

## Ausführung

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

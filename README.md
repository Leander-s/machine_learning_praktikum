# TODO

- Liste von **dependencies** für den source code. Habe ich im paper nicht gesehen.

# Code

Source code der im paper gegeben war.

### gnn.py

#### Linux
    python3 ./gnn.py ../data/<dataset>/<data>.csv <model> <task>
#### Windows
    python gnn.py ..\data\<dataset>\<data>.csv <model> <task>
(Hab windows nicht getestet)

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

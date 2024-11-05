import sys
import pandas as pd

dataset_for_modeling = pd.read_csv(sys.argv[1])
washed_dataset = pd.read_csv(sys.argv[2])

dataset = pd.concat([dataset_for_modeling, washed_dataset], axis=1)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed: |^$', regex=True)]
dataset = dataset.loc[:, ~dataset.columns.str.contains('iupac', regex=True)]
dataset = dataset.loc[:, ~dataset.columns.str.contains('smiles', regex=True)]
dataset = dataset.loc[:, ~dataset.columns.duplicated()]
dataset = dataset.loc[:, ~dataset.T.duplicated()]

dataset.to_csv("./completed_data/" + sys.argv[1].split("/")[-1].split(".")[0] + "_.csv")

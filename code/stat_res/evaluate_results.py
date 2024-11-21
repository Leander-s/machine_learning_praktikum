import sys
import numpy as np
import pandas as pd

if (len(sys.argv) == 1):
    raise Exception("Give result data as argument")

filepath = sys.argv[1]
name = filepath.split('/')[-1].split('_')[0]
model = filepath.split('/')[-1].split('_')[1]

df = pd.read_csv(filepath)

scorer = 'rmse' if 'rmse' in df.columns else 'roc_auc'

df_train = df[df['set'] == "train"].drop(columns='set')
df_val = df[df['set'] == "validation"].drop(columns='set')
df_test = df[df['set'] == "test"].drop(columns='set')

train_data = df_train[scorer].to_numpy()
val_data = df_val[scorer].to_numpy()
test_data = df_test[scorer].to_numpy()

train_std = np.std(train_data)
val_std = np.std(val_data)
test_std = np.std(test_data)

train_mean = np.mean(train_data)
val_mean = np.mean(val_data)
test_mean = np.mean(test_data)

f = open(name + "_" + model + "_results" + ".txt", "w")

f.write(f"Train : {train_mean} +- {train_std}\nValidation : {val_mean} +- {val_std}\nTest : {test_mean} +- {test_std}")

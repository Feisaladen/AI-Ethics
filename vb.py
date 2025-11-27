import pandas as pd

df = pd.read_csv("compas-scores-raw.csv")

print(df.columns.tolist())

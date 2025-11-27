import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# ----------------------------------------------------
# 1) LOAD DATA
# ----------------------------------------------------
full_df = pd.read_csv("compas-scores-raw.csv")
two_year_df = pd.read_csv("compas-scores-two-years.csv")

print("\n--- compas-scores-raw.csv ---")
print(full_df.columns.tolist())

print("\n--- compas-scores-two-years.csv ---")
print(two_year_df.columns.tolist())

# ----------------------------------------------------
# 2) SELECT DATASET TO USE
# ----------------------------------------------------
df = two_year_df.copy()  

# ----------------------------------------------------
# 3) CLEAN + PREPARE VARIABLES
# ----------------------------------------------------
# Convert race to binary: White = 1, Black/Other = 0
df['race_binary'] = df['race'].apply(lambda x: 1 if x == "Caucasian" else 0)

# Keep only numeric columns for AIF360
df_numeric = df[['race_binary', 'decile_score', 'two_year_recid']]

# Drop missing rows
df_numeric = df_numeric.dropna(subset=['race_binary', 'decile_score', 'two_year_recid'])

# ----------------------------------------------------
# 4) CREATE AIF360 DATASET
# ----------------------------------------------------
dataset = BinaryLabelDataset(
    df=df_numeric,
    label_names=['two_year_recid'],
    protected_attribute_names=['race_binary'],
    favorable_label=0,     # did NOT recidivate
    unfavorable_label=1    # DID recidivate
)

# ----------------------------------------------------
# 5) SPLIT BY RACE (Corrected)
# ----------------------------------------------------
# Get the column index for race_binary
race_idx = dataset.protected_attribute_names.index('race_binary')

# White (privileged)
white_mask = dataset.features[:, race_idx] == 1
white = dataset.subset(np.where(white_mask)[0])

# Black (unprivileged)
black_mask = dataset.features[:, race_idx] == 0
black = dataset.subset(np.where(black_mask)[0])

# ----------------------------------------------------
# 6) RECIDIVISM RATES
# ----------------------------------------------------
white_recid = np.mean(white.labels)
black_recid = np.mean(black.labels)
print("\n---- RECIDIVISM RATES ----")
print("White Recidivism Rate:", white_recid)
print("Black Recidivism Rate:", black_recid)

# Average risk scores
white_risk = df_numeric[df_numeric['race_binary'] == 1]['decile_score'].mean()
black_risk = df_numeric[df_numeric['race_binary'] == 0]['decile_score'].mean()

print("\n---- AVERAGE RISK SCORES ----")
print("Average White Risk Score:", white_risk)
print("Average Black Risk Score:", black_risk)

# ----------------------------------------------------
# 7) FAIRNESS METRICS
# ----------------------------------------------------
metric = ClassificationMetric(
    dataset, dataset,
    privileged_groups=[{'race_binary': 1}],  # White
    unprivileged_groups=[{'race_binary': 0}] # Black
)

print("\n---- FAIRNESS METRICS ----")
print("Statistical Parity Difference:", metric.statistical_parity_difference())
print("Disparate Impact:", metric.disparate_impact())
print("Equal Opportunity Difference:", metric.equal_opportunity_difference())
print("Average Odds Difference:", metric.average_odds_difference())
print("False Positive Rate Difference:", metric.false_positive_rate_difference())

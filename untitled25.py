# -*- coding -*-
"""Untitled25.ipynb

It is also Available on Colab.

Original file is located at
    https://colab.research.google.com/drive/1BuAR3v_TFzvvz2LVcnRVwSYzw6Uh_bT4
"""

"""
============================================================
COUNTERFACTUAL EXPLANATION PIPELINE
Based on Rubin (1974) + Structural Causal Models
============================================================

This code demonstrates:
1. Potential Outcomes (Y0, Y1)
2. Individual Treatment Effects (ITE)
3. Average Treatment Effects (ATE, ATT)
4. SCM mediation (ACDE, ACME)
5. Counterfactual prediction
6. Graphical explanations

Dataset: counterfactual_dataset_1000.csv
============================================================
"""

# ------------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("counterfactual_dataset_1000.csv")

# ------------------------------------------------------------
# POTENTIAL OUTCOMES FRAMEWORK (Rubin)
# ------------------------------------------------------------
# Individual Treatment Effect
df["ITE"] = df["Y_1"] - df["Y_0"]

ATE = df["ITE"].mean()
ATT = df[df["treatment"] == 1]["ITE"].mean()

print("Average Treatment Effect (ATE):", ATE)
print("Average Treatment Effect on Treated (ATT):", ATT)

# ------------------------------------------------------------
# STRUCTURAL CAUSAL MODEL (SCM)
# Mediator Model: M = f(T, X)
# Outcome Model:  Y = f(T, M, X)
# ------------------------------------------------------------
X_m = sm.add_constant(df[["treatment", "age", "income", "education_years"]])
mediator_model = sm.OLS(df["mediator"], X_m).fit()

X_y = sm.add_constant(df[["treatment", "mediator", "age", "income", "education_years"]])
outcome_model = sm.OLS(df["outcome"], X_y).fit()

# Direct and Indirect Effects
ACDE = outcome_model.params["treatment"]
ACME = mediator_model.params["treatment"] * outcome_model.params["mediator"]

print("Average Controlled Direct Effect (ACDE):", ACDE)
print("Average Causal Mediation Effect (ACME):", ACME)

# ------------------------------------------------------------
# GRAPH 1: DISTRIBUTION OF INDIVIDUAL TREATMENT EFFECTS
# ------------------------------------------------------------
plt.figure()
plt.hist(df["ITE"], bins=30)
plt.xlabel("Individual Treatment Effect (Y1 − Y0)")
plt.ylabel("Frequency")
plt.title("Distribution of Individual Treatment Effects")
plt.show()

# ------------------------------------------------------------
# GRAPH 2: EFFECT OF TREATMENT ON MEDIATOR
# ------------------------------------------------------------
plt.figure()
plt.scatter(df["treatment"], df["mediator"])
plt.xlabel("Treatment")
plt.ylabel("Mediator")
plt.title("Treatment → Mediator Relationship")
plt.show()

# ------------------------------------------------------------
# GRAPH 3: EFFECT OF MEDIATOR ON OUTCOME
# ------------------------------------------------------------
plt.figure()
plt.scatter(df["mediator"], df["outcome"])
plt.xlabel("Mediator")
plt.ylabel("Outcome")
plt.title("Mediator → Outcome Relationship")
plt.show()

# ------------------------------------------------------------
# GRAPH 4: COUNTERFACTUAL EXPLANATION (Single Individual)
# ------------------------------------------------------------
i = df.iloc[0]

# Flip treatment
counterfactual_treatment = 1 - i["treatment"]

# Counterfactual mediator
mediator_cf = (
    mediator_model.params["const"]
    + mediator_model.params["treatment"] * counterfactual_treatment
    + mediator_model.params["age"] * i["age"]
    + mediator_model.params["income"] * i["income"]
    + mediator_model.params["education_years"] * i["education_years"]
)

# Counterfactual outcome
outcome_cf = (
    outcome_model.params["const"]
    + outcome_model.params["treatment"] * counterfactual_treatment
    + outcome_model.params["mediator"] * mediator_cf
    + outcome_model.params["age"] * i["age"]
    + outcome_model.params["income"] * i["income"]
    + outcome_model.params["education_years"] * i["education_years"]
)

plt.figure()
plt.bar(
    ["Observed Outcome", "Counterfactual Outcome"],
    [i["outcome"], outcome_cf]
)
plt.ylabel("Outcome Value")
plt.title("Observed vs Counterfactual Outcome")
plt.show()

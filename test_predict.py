# -*- coding: Test -*-

It is also run on Colab.

Original file is located at
    https://colab.research.google.com/drive/1qiAI2jrTARQd_a9yQIFYqOu7Gw4CX7Q3
"""

# -*- coding: Test case -*-
"""
============================================================
COUNTERFACTUAL EXPLANATION PIPELINE
Train + Test Inference
Based on Rubin (1974) + Structural Causal Models
============================================================

This code demonstrates:
1. Potential Outcomes (Y0, Y1)
2. Individual Treatment Effects (ITE)
3. Average Treatment Effects (ATE, ATT)
4. SCM mediation (ACDE, ACME)
5. Test data counterfactual inference
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
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("counterfactual_dataset_1000.csv")

# ------------------------------------------------------------
# POTENTIAL OUTCOMES FRAMEWORK (GROUND TRUTH)
# ------------------------------------------------------------
df["ITE"] = df["Y_1"] - df["Y_0"]

ATE = df["ITE"].mean()
ATT = df[df["treatment"] == 1]["ITE"].mean()

print("Average Treatment Effect (ATE):", ATE)
print("Average Treatment Effect on Treated (ATT):", ATT)

# ------------------------------------------------------------
# TRAIN–TEST SPLIT
# ------------------------------------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ------------------------------------------------------------
# STRUCTURAL CAUSAL MODEL (TRAINING)
# Mediator: M = f(T, X)
# Outcome:  Y = f(T, M, X)
# ------------------------------------------------------------
X_m_train = sm.add_constant(
    train_df[["treatment", "age", "income", "education_years"]]
)
mediator_model = sm.OLS(train_df["mediator"], X_m_train).fit()

X_y_train = sm.add_constant(
    train_df[["treatment", "mediator", "age", "income", "education_years"]]
)
outcome_model = sm.OLS(train_df["outcome"], X_y_train).fit()

# Direct and Indirect Effects
ACDE = outcome_model.params["treatment"]
ACME = mediator_model.params["treatment"] * outcome_model.params["mediator"]

print("Average Controlled Direct Effect (ACDE):", ACDE)
print("Average Causal Mediation Effect (ACME):", ACME)

# ------------------------------------------------------------
# TEST DATA — FACTUAL INFERENCE
# ------------------------------------------------------------
X_m_test = sm.add_constant(
    test_df[["treatment", "age", "income", "education_years"]]
)
test_df["mediator_pred"] = mediator_model.predict(X_m_test)

X_y_test = sm.add_constant(
    pd.concat(
        [
            test_df[["treatment"]],
            test_df[["mediator_pred"]].rename(columns={"mediator_pred": "mediator"}),
            test_df[["age", "income", "education_years"]],
        ],
        axis=1,
    )
)
test_df["outcome_pred"] = outcome_model.predict(X_y_test)

# ------------------------------------------------------------
# TEST DATA — COUNTERFACTUAL INFERENCE
# ------------------------------------------------------------
# Flip treatment
test_df["treatment_cf"] = 1 - test_df["treatment"]

# Counterfactual mediator
X_m_cf = sm.add_constant(
    pd.concat(
        [
            test_df[["treatment_cf"]].rename(columns={"treatment_cf": "treatment"}),
            test_df[["age", "income", "education_years"]],
        ],
        axis=1,
    )
)
test_df["mediator_cf"] = mediator_model.predict(X_m_cf)

# Counterfactual outcome
X_y_cf = sm.add_constant(
    pd.concat(
        [
            test_df[["treatment_cf"]].rename(columns={"treatment_cf": "treatment"}),
            test_df[["mediator_cf"]].rename(columns={"mediator_cf": "mediator"}),
            test_df[["age", "income", "education_years"]],
        ],
        axis=1,
    )
)
test_df["outcome_cf"] = outcome_model.predict(X_y_cf)

# ------------------------------------------------------------
# TEST SET INDIVIDUAL TREATMENT EFFECT
# ------------------------------------------------------------
test_df["ITE_test"] = test_df["outcome_cf"] - test_df["outcome_pred"]

print("\nTest-set ITE (first 5 rows):")
print(test_df[["outcome_pred", "outcome_cf", "ITE_test"]].head())

# ------------------------------------------------------------
# GRAPH 1: DISTRIBUTION OF TRUE ITE
# ------------------------------------------------------------
plt.figure()
plt.hist(df["ITE"], bins=30)
plt.xlabel("Individual Treatment Effect (Y1 − Y0)")
plt.ylabel("Frequency")
plt.title("Distribution of True Individual Treatment Effects")
plt.show()

# ------------------------------------------------------------
# GRAPH 2: TEST-SET ITE DISTRIBUTION
# ------------------------------------------------------------
plt.figure()
plt.hist(test_df["ITE_test"], bins=30)
plt.xlabel("Estimated Test-set ITE")
plt.ylabel("Frequency")
plt.title("Distribution of Test-set Counterfactual Effects")
plt.show()

# ------------------------------------------------------------
# GRAPH 3: TREATMENT → MEDIATOR
# ------------------------------------------------------------
plt.figure()
plt.scatter(df["treatment"], df["mediator"])
plt.xlabel("Treatment")
plt.ylabel("Mediator")
plt.title("Treatment → Mediator Relationship")
plt.show()

# ------------------------------------------------------------
# GRAPH 4: MEDIATOR → OUTCOME
# ------------------------------------------------------------
plt.figure()
plt.scatter(df["mediator"], df["outcome"])
plt.xlabel("Mediator")
plt.ylabel("Outcome")
plt.title("Mediator → Outcome Relationship")
plt.show()

# ------------------------------------------------------------
# GRAPH 5: SINGLE TEST INDIVIDUAL COUNTERFACTUAL
# ------------------------------------------------------------
i = test_df.iloc[0]

plt.figure()
plt.bar(
    ["Observed (Predicted)", "Counterfactual"],
    [i["outcome_pred"], i["outcome_cf"]],
)
plt.ylabel("Outcome Value")
plt.title("Observed vs Counterfactual Outcome (Test Individual)")
plt.show()
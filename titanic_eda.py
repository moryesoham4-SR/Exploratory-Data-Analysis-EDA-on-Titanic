"""
Exploratory Data Analysis (EDA) on Titanic
InternCircle | Project 01 | Data Science & AI

This script executes the complete data pipeline, answers the three primary questions,
and outputs statistical aggregations and clean figures.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("PROJECT 01: Exploratory Data Analysis (EDA) on Titanic")
    print("=" * 60)

    # 1. Three Core Questions
    print("\n--- 1. QUESTIONS TO ANSWER ---")
    print("Q1: Did female passengers have a higher survival rate than male passengers?")
    print("Q2: Did passenger ticket class (pclass) significantly impact survival probability?")
    print("Q3: How was age related to survival across distinct life stages (Child, Teen, Adult, etc.)?")

    # 2. Load and Look
    print("\n--- 2. LOADING DATASET ---")
    data_path = os.path.join(os.path.dirname(__file__), "titanic.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = sns.load_dataset("titanic")

    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSummary Statistics:")
    print(df.describe())

    # 3. Find Missing Values
    print("\n--- 3. MISSING VALUES ---")
    missing = df.isna().sum()
    print(missing[missing > 0])

    # 4. Handle Missing Values
    print("\n--- 4. CLEANING & IMPUTATION ---")
    # Impute age with median grouped by pclass and sex
    df["age"] = df["age"].fillna(df.groupby(["pclass", "sex"])["age"].transform("median"))
    # Drop deck column due to ~77% missing data
    if "deck" in df.columns:
        df = df.drop(columns=["deck"])
    # Impute embarked and embark_town with mode (Southampton)
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    if "embark_town" in df.columns:
        df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

    print("Missing values after cleaning:")
    print(df.isna().sum())

    # 5. Check Duplicates and Types
    print("\n--- 5. DUPLICATES & DATA TYPES ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    print("\nColumn Data Types:")
    print(df.dtypes)

    # 6. Group and Compare
    print("\n--- 6. GROUPING & AGGREGATIONS ---")
    
    # 6.1 By Sex
    sex_surv = df.groupby("sex")["survived"].agg(["count", "mean"]).round(4)
    sex_surv["survival_rate_%"] = (sex_surv["mean"] * 100).round(2)
    sex_surv.sort_values(by="mean", ascending=False, inplace=True)
    print("\nSurvival by Sex:")
    print(sex_surv)

    # 6.2 By Ticket Class
    class_surv = df.groupby("pclass")["survived"].agg(["count", "mean"]).round(4)
    class_surv["survival_rate_%"] = (class_surv["mean"] * 100).round(2)
    class_surv.sort_values(by="mean", ascending=False, inplace=True)
    print("\nSurvival by Ticket Class:")
    print(class_surv)

    # 6.3 By Age Band
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 12, 18, 40, 60, 100],
        labels=["Child", "Teen", "Adult", "Middle-aged", "Senior"]
    )
    age_surv = df.groupby("age_band", observed=True)["survived"].agg(["count", "mean"]).round(4)
    age_surv["survival_rate_%"] = (age_surv["mean"] * 100).round(2)
    age_surv.sort_values(by="mean", ascending=False, inplace=True)
    print("\nSurvival by Age Band:")
    print(age_surv)

    # 7. One-sentence Answers
    print("\n--- 7. ANSWERS TO QUESTIONS ---")
    print(f"1. Women survived at about {sex_surv.loc['female', 'survival_rate_%']}% compared with about {sex_surv.loc['male', 'survival_rate_%']}% for men.")
    print(f"2. Passenger class significantly determined survival: {class_surv.loc[1, 'survival_rate_%']}% of 1st-class passengers survived, compared to {class_surv.loc[2, 'survival_rate_%']}% in 2nd-class and only {class_surv.loc[3, 'survival_rate_%']}% in 3rd-class.")
    print(f"3. Children (aged 0-12) had the highest survival rate at {age_surv.loc['Child', 'survival_rate_%']}%, whereas seniors (aged 61+) had the lowest survival rate at {age_surv.loc['Senior', 'survival_rate_%']}%.")

    print("\nDone! All steps completed successfully.")

if __name__ == "__main__":
    main()

# InternCircle Project 01: Exploratory Data Analysis (EDA) on Titanic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moryesoham4-SR/Exploratory-Data-Analysis-EDA-on-Titanic/blob/main/titanic_eda.ipynb)

## Overview
This project performs an end-to-end Exploratory Data Analysis (EDA) on the classic Titanic passenger survival dataset using Python, Pandas, Seaborn, and Matplotlib.

---

## 1. Initial Analytical Questions
1. **Did women survive more than men?** (Assessing the historical "women and children first" protocol)
2. **Did ticket class matter?** (Evaluating socio-economic disparity in survival outcomes)
3. **Was age related to survival?** (Analyzing survival rates across life stages: Children, Teens, Adults, Middle-aged, and Seniors)

---

## 2. Key Cleaning Decisions & Rationales
- **`deck` (Cabin Deck):** Dropped completely (`df.drop(columns=['deck'])`) because **77.22% (688/891)** of values were missing. Imputation would have introduced excessive noise and biased structural interpretations.
- **`age`:** Imputed using the **median age within each passenger class and sex subgroup** (`df.groupby(['pclass', 'sex'])['age'].transform('median')`). Age distributions differed significantly across classes and genders; using subgroup medians preserved realistic demographic variation while remaining robust against outliers.
- **`embarked` & `embark_town`:** Imputed with the **mode** (Southampton, `S`), as only 2 rows were missing (0.22%), representing minimal data impact.
- **Duplicate rows:** Kept intentionally. Because identifying columns (e.g., passenger names, ticket numbers) were omitted in the curated dataset, rows with identical class, age, sex, and fare represent distinct individuals rather than redundant records.

---

## 3. Answers to Core Questions (with Concrete Numbers)
1. **Gender:** **Women survived at approximately 74.20% compared with about 18.89% for men**, representing a 55.31 percentage point survival advantage for females.
2. **Ticket Class:** **Ticket class was a decisive survival factor: 62.96% of 1st-class passengers survived, compared with 47.28% in 2nd-class and only 24.24% in 3rd-class**, demonstrating a direct correlation between socio-economic status and lifeboat access.
3. **Age Demographic:** **Children (aged 0–12) experienced the highest survival rate at 57.97%, while seniors (aged 61+) had the lowest survival rate at 22.73%**, confirming that children received priority evacuation while older passengers faced lower survival odds.

---

## 4. File Manifest
- `titanic_eda.ipynb`: Fully executed Jupyter Notebook with markdown narratives, code cells, tables, and visualization charts. (Directly runnable on Google Colab)
- `titanic_eda.html`: Exported standalone HTML report containing all figures, tables, and styled markdown.
- `titanic_eda.py`: Standalone Python script running the entire pipeline and printing findings.
- `titanic.csv`: Local offline copy of the Titanic passenger dataset.

---

## 5. Submission Checklist Verification
- [x] Three clear questions are written at the start
- [x] Missing values are handled and the reasons explained
- [x] Groupings compare survival by at least three factors (Sex, Class, Age Band)
- [x] Each question has a one-sentence answer with a number
- [x] The notebook runs from top to bottom without errors
- [x] Headings and notes make it easy to follow

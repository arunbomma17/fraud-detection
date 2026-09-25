"""Script to generate all EDA figures for reports and Streamlit dashboard."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COL, AMOUNT_COL, TIME_COL
from src.data_loader import load_data

sns.set_theme(style="whitegrid")
df = load_data()

# 1. Class Distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
counts = df[TARGET_COL].value_counts()
sns.countplot(data=df, x=TARGET_COL, palette=["#3B82F6", "#EF4444"], ax=ax[0])
ax[0].set_title("Class Distribution (Log Scale)", weight="bold")
ax[0].set_yscale("log")
ax[0].set_xticklabels(["Genuine", "Fraud"])
ax[1].pie(counts, labels=["Genuine", "Fraud"], autopct="%1.2f%%", colors=["#3B82F6", "#EF4444"], explode=[0, 0.15])
ax[1].set_title("Fraud vs Genuine Percentage", weight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
plt.close()

# 2. Amount Distribution
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df[df[TARGET_COL] == 0][AMOUNT_COL], bins=40, color="#3B82F6", ax=ax[0], label="Genuine")
sns.histplot(df[df[TARGET_COL] == 1][AMOUNT_COL], bins=40, color="#EF4444", ax=ax[0], label="Fraud")
ax[0].set_title("Amount Distribution (Linear)", weight="bold")
ax[0].set_xlim([0, 1000])
ax[0].legend()

sns.histplot(np.log1p(df[df[TARGET_COL] == 0][AMOUNT_COL]), bins=40, color="#3B82F6", ax=ax[1], label="Genuine")
sns.histplot(np.log1p(df[df[TARGET_COL] == 1][AMOUNT_COL]), bins=40, color="#EF4444", ax=ax[1], label="Fraud")
ax[1].set_title("Log(Amount + 1) Distribution", weight="bold")
ax[1].legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "amount_distribution.png", dpi=300)
plt.close()

# 3. Time Distribution
df["hour_of_day"] = (df[TIME_COL] / 3600.0) % 24.0
plt.figure(figsize=(9, 4))
sns.kdeplot(df[df[TARGET_COL] == 0]["hour_of_day"], color="#3B82F6", label="Genuine", fill=True, alpha=0.3)
sns.kdeplot(df[df[TARGET_COL] == 1]["hour_of_day"], color="#EF4444", label="Fraud", fill=True, alpha=0.3)
plt.title("Transaction Activity by Hour of Day", weight="bold")
plt.xlabel("Hour (0-23)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "time_distribution.png", dpi=300)
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 4.5))
corr = df.corr()[TARGET_COL].drop(TARGET_COL).sort_values()
colors = ["#EF4444" if c < 0 else "#3B82F6" for c in corr]
corr.plot(kind="bar", color=colors)
plt.title("Feature Correlations with Fraud Label (Class)", weight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# 5. Feature Distributions (V14, V12, V17, V4)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()
for i, feat in enumerate(["V14", "V12", "V17", "V4"]):
    sns.boxplot(data=df, x=TARGET_COL, y=feat, palette=["#3B82F6", "#EF4444"], ax=axes[i])
    axes[i].set_title(f"Distribution of {feat} by Class", weight="bold")
    axes[i].set_xticklabels(["Genuine", "Fraud"])
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_distributions.png", dpi=300)
plt.close()

print("All EDA figures generated successfully in:", FIGURES_DIR)

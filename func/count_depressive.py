import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o CSV (substitua 'arquivo.csv' pelo nome do seu arquivo)
df = pd.read_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset/dataset_final_3003.csv")

# Define the column containing item types (replace 'item_type' with the actual column name)
column_name = "depressive"

# # Count the frequency of each item type
count_data = df[column_name].value_counts()
print(count_data)

# Set scientific publication style
sns.set_theme(style="whitegrid", palette="muted")

# ====== Histogram (Barplot) ======
plt.figure(figsize=(10, 6))
sns.barplot(x=count_data.index, y=count_data.values, color="royalblue")

# Improve visualization
plt.xlabel("Item Type", fontsize=14, fontweight="bold")
plt.ylabel("Frequency", fontsize=14, fontweight="bold")
plt.title("Frequency Distribution of Item Types", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Remove unnecessary borders
sns.despine()

# Save the histogram
plt.savefig("frequency_histogram.png", dpi=300, bbox_inches="tight")
plt.show()


# ====== Boxplot ======
plt.figure(figsize=(6, 6))
sns.boxplot(y=count_data.values, color="darkorange")

# Improve visualization
plt.ylabel("Frequency", fontsize=14, fontweight="bold")
plt.title("Boxplot of Item Type Frequencies", fontsize=16, fontweight="bold")

# Remove unnecessary borders
sns.despine()

# Save the boxplot
plt.savefig("frequency_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()
import pandas as pd

df = pd.read_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset_inicial.csv")

df_cleaned = df.drop(columns=["repostCount", "replyCount", "link", "image", "createdAt"])

df_cleaned = df_cleaned.drop_duplicates(subset=["text"])

df_cleaned.to_csv("novo_dataset.csv", index=False)

depressive_count = df_cleaned["depressive"].value_counts()

print(depressive_count)
import pandas as pd

df = pd.read_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset_inicial.csv")

df_cleaned = df.drop(columns=["repostCount", "replyCount", "link", "image", "createdAt"])

df_cleaned = df_cleaned.drop_duplicates(subset=["text"])

save_path = "/scratch/gabriel.lemos/Bluesky-Depression/scrapers"

df_cleaned.to_csv(f"{save_path}dataset_limpo.csv", index=False)

depressive_count = df_cleaned["depressive"].value_counts()

depressive_count.to_csv(f"{save_path}depressive_count.csv", index=True)

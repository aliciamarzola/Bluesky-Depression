import pandas as pd

df1 = pd.read_csv("dataset_1_final.csv")
df2 = pd.read_csv("dataset_2_final.csv")

df2 = df2.drop_duplicates(subset=["text"])
df2 = df2.loc[df2['depressive']==1]
df2['depressive'] = df2['depressive'].astype('Int64')

df1 = df1.drop(columns=["repostCount", "replyCount", "link", "image", "createdAt"])
df1['depressive'] = df1['depressive'].astype(str).str.strip()
df1 = df1[df1['depressive'].isin(['0', '1'])]
df1['depressive'] = df1['depressive'].astype('Int64')

df1.to_csv(f"dataset-characterization.csv", index=False)

df_concat = pd.concat([df1, df2], axis=0)
df_concat = df_concat[['text', 'depressive']]
print(df_concat.head())

import emoji

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='') if isinstance(text, str) else text

df_concat['text'] = df_concat['text'].apply(remove_emojis)

df_concat.to_csv(f"dataset_final.csv", index=False)

depressive_count = df_concat["depressive"].value_counts()

depressive_count.to_csv(f"depressive_count.csv", index=True)

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import emoji

def remove_emojis(text):
    return emoji.replace_emoji(text, "")
# Carregar dataset
df = pd.read_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset2_limpo.csv")
df["text"] = df["text"].astype(str).fillna("")
df["text"] = df["text"].apply(remove_emojis)  # Remover emojis
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
df = df[df['text'].str.strip() != '']
df = df[df['depressive'] == 1]  # Manter apenas os depressivos

import pandas as pd
import emoji

import torch

if torch.cuda.is_available():
    print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
else:
    print("Nenhuma GPU disponível, rodando na CPU.")


""" # Carregar o dataset CSV
df = pd.read_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset_final.csv")

# Função para remover emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='') if isinstance(text, str) else text

# Aplicar a função a uma coluna específica (substitua 'coluna' pelo nome correto)
df['text'] = df['text'].apply(remove_emojis)

# Salvar o CSV limpo
df.to_csv("dataset_final_f_emoji2.csv", index=False)
 """